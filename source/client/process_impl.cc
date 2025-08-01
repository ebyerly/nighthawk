#include "source/client/process_impl.h"

#include <sys/file.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "envoy/common/optref.h"
#include "envoy/http/http_server_properties_cache.h"
#include "envoy/network/address.h"
#include "envoy/server/filter_config.h"
#include "envoy/stats/sink.h"
#include "envoy/stats/store.h"

#include "nighthawk/client/output_collector.h"
#include "nighthawk/common/factories.h"
#include "nighthawk/user_defined_output/user_defined_output_plugin.h"

#include "external/envoy/envoy/config/xds_manager.h"
#include "external/envoy/source/common/api/api_impl.h"
#include "external/envoy/source/common/common/cleanup.h"
#include "external/envoy/source/common/common/regex.h"
#include "external/envoy/source/common/common/statusor.h"
#include "external/envoy/source/common/config/utility.h"
#include "external/envoy/source/common/config/xds_manager_impl.h"
#include "external/envoy/source/common/event/dispatcher_impl.h"
#include "external/envoy/source/common/event/real_time_system.h"
#include "external/envoy/source/common/http/http_server_properties_cache_manager_impl.h"
#include "external/envoy/source/common/init/manager_impl.h"
#include "external/envoy/source/common/local_info/local_info_impl.h"
#include "external/envoy/source/common/network/dns_resolver/dns_factory_util.h"
#include "external/envoy/source/common/network/utility.h"
#include "external/envoy/source/common/protobuf/protobuf.h"
#include "external/envoy/source/common/runtime/runtime_impl.h"
#include "external/envoy/source/common/singleton/manager_impl.h"
#include "external/envoy/source/common/stats/tag_producer_impl.h"
#include "external/envoy/source/common/thread_local/thread_local_impl.h"
#include "external/envoy/source/server/null_overload_manager.h"
#include "external/envoy/source/server/server.h"
#include "external/envoy_api/envoy/config/core/v3/resolver.pb.h"

#include "source/client/process_bootstrap.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/optional.h"

// TODO(oschaaf): See if we can leverage a static module registration like Envoy does to avoid the
// ifdefs in this file.
#ifdef ZIPKIN_ENABLED
#include "external/envoy/source/extensions/tracers/zipkin/zipkin_tracer_impl.h"
#endif
#include "external/envoy/source/server/options_impl.h"
#include "external/envoy/source/server/options_impl_platform.h"

#include "api/client/options.pb.h"
#include "api/client/output.pb.h"

#include "source/common/frequency.h"
#include "source/common/uri_impl.h"
#include "source/common/utility.h"

#include "source/client/benchmark_client_impl.h"
#include "source/client/client.h"
#include "source/client/client_worker_impl.h"
#include "source/client/factories_impl.h"
#include "source/client/options_impl.h"
#include "source/client/sni_utility.h"

#include "source/user_defined_output/user_defined_output_plugin_creator.h"

using namespace std::chrono_literals;

namespace Nighthawk {
namespace Client {
namespace {

using ::envoy::config::bootstrap::v3::Bootstrap;
using ::envoy::config::core::v3::TypedExtensionConfig;

// Helps in generating a bootstrap for the process.
// This is a class only to allow the use of the ENVOY_LOG macros.
class BootstrapFactory : public Envoy::Logger::Loggable<Envoy::Logger::Id::main> {
public:
  // Determines the concurrency Nighthawk should use based on configuration
  // (options) and the available machine resources.
  static uint32_t determineConcurrency(const Options& options) {
    uint32_t cpu_cores_with_affinity = Envoy::OptionsImplPlatform::getCpuCount();
    bool autoscale = options.concurrency() == "auto";
    // TODO(oschaaf): Maybe, in the case where the concurrency flag is left out, but
    // affinity is set / we don't have affinity with all cores, we should default to autoscale.
    // (e.g. we are called via taskset).
    uint32_t concurrency = autoscale ? cpu_cores_with_affinity : std::stoi(options.concurrency());

    if (autoscale) {
      ENVOY_LOG(info, "Detected {} (v)CPUs with affinity..", cpu_cores_with_affinity);
    }
    std::string duration_as_string =
        options.noDuration() ? "No time limit"
                             : fmt::format("Time limit: {} seconds", options.duration().count());
    ENVOY_LOG(info, "Starting {} threads / event loops. {}.", concurrency, duration_as_string);
    ENVOY_LOG(info, "Global targets: {} connections and {} calls per second.",
              options.connections() * concurrency, options.requestsPerSecond() * concurrency);

    if (concurrency > 1) {
      ENVOY_LOG(info, "   (Per-worker targets: {} connections and {} calls per second)",
                options.connections(), options.requestsPerSecond());
    }

    return concurrency;
  }
};

// Implementation of dummy StatsConfig.
class StatsConfigImpl : public Envoy::Server::Configuration::StatsConfig {
public:
  StatsConfigImpl() : flush_interval_(std::chrono::seconds(5)) {};

  const std::list<Envoy::Stats::SinkPtr>& sinks() const override { return sinks_; }
  std::chrono::milliseconds flushInterval() const override { return flush_interval_; }
  bool flushOnAdmin() const override { return false; }

  void addSink(Envoy::Stats::SinkPtr sink) { sinks_.emplace_back(std::move(sink)); }

  bool enableDeferredCreationStats() const override { return false; }

private:
  std::list<Envoy::Stats::SinkPtr> sinks_;
  const std::chrono::milliseconds flush_interval_;
};

// A fake ServerLifecycleNotifier. Because it does nothing, it's safe to just create
// multiple instances of it, rather than manage the lifetime of a single one.
class NighthawkLifecycleNotifierImpl : public Envoy::Server::ServerLifecycleNotifier {
public:
  HandlePtr registerCallback(Stage, StageCallback) override {
    PANIC(
        "NighthawkLifecycleNotifierImpl::registerCallbacki(Stage, StageCallback) not implemented");
  }
  HandlePtr registerCallback(Stage, StageCallbackWithCompletion) override {
    PANIC("NighthawkLifecycleNotifierImpl::registerCallback(Stage, StageCallbackWithCompletion) "
          "not implemented");
  }
};

// Implementation of Envoy::Server::Configuration::ServerFactoryContext.
class NighthawkServerFactoryContext : public Envoy::Server::Configuration::ServerFactoryContext {
public:
  explicit NighthawkServerFactoryContext(Envoy::Server::Instance& server)
      : server_(server), server_scope_(server_.stats().createScope("")) {}

  const Envoy::Server::Options& options() override { return server_.options(); };

  Envoy::Event::Dispatcher& mainThreadDispatcher() override { return server_.dispatcher(); }

  Envoy::Api::Api& api() override { return server_.api(); }

  Envoy::LocalInfo::LocalInfo& localInfo() const override { return server_.localInfo(); }

  Envoy::OptRef<Envoy::Server::Admin> admin() override { return server_.admin(); }

  Envoy::Runtime::Loader& runtime() override { return server_.runtime(); }

  Envoy::Singleton::Manager& singletonManager() override { return server_.singletonManager(); }

  Envoy::ProtobufMessage::ValidationVisitor& messageValidationVisitor() override {
    return Envoy::ProtobufMessage::getStrictValidationVisitor();
  };

  Envoy::Stats::Scope& scope() override {
    PANIC("NighthawkServerFactoryContext::scope not implemented");
  };

  Envoy::Stats::Scope& serverScope() override { return *server_scope_; };

  Envoy::ThreadLocal::Instance& threadLocal() override { return server_.threadLocal(); }

  Envoy::Upstream::ClusterManager& clusterManager() override {
    if (cluster_manager_ != nullptr) {
      return *cluster_manager_;
    }
    PANIC("NighthawkServerFactoryContext::clusterManager not implemented");
  };

  Envoy::Config::XdsManager& xdsManager() override { return server_.xdsManager(); };

  Envoy::Http::HttpServerPropertiesCacheManager& httpServerPropertiesCacheManager() override {
    return server_.httpServerPropertiesCacheManager();
  }

  Envoy::ProtobufMessage::ValidationContext& messageValidationContext() override {
    return server_.messageValidationContext();
  };

  Envoy::TimeSource& timeSource() override { return api().timeSource(); };

  Envoy::AccessLog::AccessLogManager& accessLogManager() override {
    return server_.accessLogManager();
  }

  Envoy::Server::ServerLifecycleNotifier& lifecycleNotifier() override {
    return lifecycle_notifier_;
  }

  Envoy::Regex::Engine& regexEngine() override { return regex_engine_; }

  Envoy::Init::Manager& initManager() override {
    PANIC("NighthawkServerFactoryContext::initManager not implemented");
  };

  Envoy::Grpc::Context& grpcContext() override { return server_.grpcContext(); };

  Envoy::Router::Context& routerContext() override { return server_.routerContext(); };

  Envoy::ProcessContextOptRef processContext() override {
    PANIC("NighthawkServerFactoryContext::processContext not implemented");
  }

  Envoy::Server::DrainManager& drainManager() override {
    PANIC("NighthawkServerFactoryContext::drainManager not implemented");
  };

  Envoy::Ssl::ContextManager& sslContextManager() override {
    if (ssl_context_manager_ != nullptr) {
      return *ssl_context_manager_;
    }
    PANIC("NighthawkServerFactoryContext::sslContextManager not implemented");
  }

  Envoy::Secret::SecretManager& secretManager() override { return server_.secretManager(); }

  Envoy::Server::Configuration::StatsConfig& statsConfig() override { return stats_config_; }

  envoy::config::bootstrap::v3::Bootstrap& bootstrap() override {
    PANIC("NighthawkServerFactoryContext::bootstrap not implemented");
  }

  Envoy::Http::Context& httpContext() override { return server_.httpContext(); }

  Envoy::Server::OverloadManager& overloadManager() override { return server_.overloadManager(); }

  Envoy::Server::OverloadManager& nullOverloadManager() override {
    PANIC("NighthawkServerFactoryContext::nullOverloadManager not implemented");
  }

  bool healthCheckFailed() const override {
    PANIC("NighthawkServerFactoryContext::healthCheckFailed not implemented");
  }

  void setClusterManager(Envoy::Upstream::ClusterManager& cluster_manager) {
    cluster_manager_ = &cluster_manager;
  }

  void setSslContextManager(Envoy::Ssl::ContextManager& ssl_context_manager) {
    ssl_context_manager_ = &ssl_context_manager;
  }

private:
  Envoy::Ssl::ContextManager* ssl_context_manager_ = nullptr;
  Envoy::Upstream::ClusterManager* cluster_manager_ = nullptr;
  Envoy::Server::Instance& server_;
  Envoy::Stats::ScopeSharedPtr server_scope_;
  StatsConfigImpl stats_config_;                      // Using the object created here.
  NighthawkLifecycleNotifierImpl lifecycle_notifier_; // A no-op object that lives here.
  Envoy::Regex::GoogleReEngine regex_engine_;         // Using the object created here.
};

// Implementation of Envoy::Server::Instance. Only methods used by Envoy's code
// when Nighthawk is running are implemented.
class NighthawkServerInstance : public Envoy::Server::Instance {
public:
  NighthawkServerInstance(Envoy::OptRef<Envoy::Server::Admin> admin, Envoy::Api::Api& api,
                          Envoy::Event::Dispatcher& dispatcher,
                          Envoy::AccessLog::AccessLogManager& log_manager,
                          Envoy::Server::Options& options, Envoy::Runtime::Loader& runtime,
                          Envoy::Singleton::Manager& singleton_manager,
                          Envoy::ThreadLocal::Instance& tls,
                          Envoy::LocalInfo::LocalInfo& local_info,
                          Envoy::ProtobufMessage::ProdValidationContextImpl& validation_context,
                          Envoy::Grpc::Context& grpc_context, Envoy::Http::Context& http_context,
                          Envoy::Router::Context& router_context, Envoy::Stats::StoreRoot& store,
                          Envoy::Secret::SecretManagerImpl& secret_manager)
      : admin_(admin), api_(api), dispatcher_(dispatcher), log_manager_(log_manager),
        options_(options), runtime_(runtime), singleton_manager_(singleton_manager),
        stats_store_(store), tls_(tls), local_info_(local_info),
        validation_context_(validation_context), grpc_context_(grpc_context),
        http_context_(http_context), router_context_(router_context),
        server_factory_context_(*this),
        http_server_properties_cache_manager_(
            server_factory_context_, Envoy::ProtobufMessage::getStrictValidationVisitor(), tls),
        xds_manager_(dispatcher, api, store, local_info, validation_context_, *this),
        secret_manager_(secret_manager),
        null_overload_manager_(std::make_unique<Envoy::Server::NullOverloadManager>(tls, false)) {}

  void run() override { PANIC("NighthawkServerInstance::run not implemented"); }
  Envoy::OptRef<Envoy::Server::Admin> admin() override { return admin_; }
  Envoy::Api::Api& api() override { return api_; }
  Envoy::Upstream::ClusterManager& clusterManager() override {
    PANIC("NighthawkServerInstance::clusterManager not implemented");
  }
  Envoy::Http::HttpServerPropertiesCacheManager& httpServerPropertiesCacheManager() override {
    return http_server_properties_cache_manager_;
  }
  const Envoy::Upstream::ClusterManager& clusterManager() const override {
    PANIC("NighthawkServerInstance::clusterManager not implemented");
  }
  Envoy::Ssl::ContextManager& sslContextManager() override {
    PANIC("NighthawkServerInstance::sslContextManager not implemented");
  }
  Envoy::Event::Dispatcher& dispatcher() override { return dispatcher_; }
  Envoy::Network::DnsResolverSharedPtr dnsResolver() override {
    PANIC("NighthawkServerInstance::dnsResolver not implemented");
  }
  void drainListeners(Envoy::OptRef<const Envoy::Network::ExtraShutdownListenerOptions>) override {
    PANIC("NighthawkServerInstance::drainListeners not implemented");
  }
  Envoy::Server::DrainManager& drainManager() override {
    PANIC("NighthawkServerInstance::drainManager not implemented");
  }
  Envoy::AccessLog::AccessLogManager& accessLogManager() override { return log_manager_; }
  void failHealthcheck(bool) override {
    PANIC("NighthawkServerInstance::failHealthcheck not implemented");
  }
  bool healthCheckFailed() override {
    PANIC("NighthawkServerInstance::healthCheckFailed not implemented");
  }
  Envoy::Server::HotRestart& hotRestart() override {
    PANIC("NighthawkServerInstance::hotRestart not implemented");
  }
  Envoy::Init::Manager& initManager() override {
    PANIC("NighthawkServerInstance::initManager not implemented");
  }
  Envoy::Server::ListenerManager& listenerManager() override {
    PANIC("NighthawkServerInstance::listenerManager not implemented");
  }
  Envoy::MutexTracer* mutexTracer() override {
    PANIC("NighthawkServerInstance::mutexTracer not implemented");
  }
  Envoy::Server::OverloadManager& overloadManager() override { return *null_overload_manager_; }
  Envoy::Server::OverloadManager& nullOverloadManager() override { return *null_overload_manager_; }
  Envoy::Secret::SecretManager& secretManager() override { return secret_manager_; }
  const Envoy::Server::Options& options() override { return options_; }
  Envoy::Runtime::Loader& runtime() override { return runtime_; }
  Envoy::Server::ServerLifecycleNotifier& lifecycleNotifier() override {
    return lifecycle_notifier_;
  }
  void shutdown() override { PANIC("NighthawkServerInstance::shutdown not implemented"); }
  bool isShutdown() override { PANIC("NighthawkServerInstance::isShutdown not implemented"); }
  void shutdownAdmin() override { PANIC("NighthawkServerInstance::shutdownAdmin not implemented"); }
  Envoy::Singleton::Manager& singletonManager() override { return singleton_manager_; }
  time_t startTimeCurrentEpoch() override {
    PANIC("NighthawkServerInstance::startTimeCurrentEpoch not implemented");
  }
  time_t startTimeFirstEpoch() override {
    PANIC("NighthawkServerInstance::startTimeFirstEpoch not implemented");
  }
  Envoy::Stats::Store& stats() override { return stats_store_; }
  Envoy::Grpc::Context& grpcContext() override { return grpc_context_; }
  Envoy::Http::Context& httpContext() override { return http_context_; }
  Envoy::Router::Context& routerContext() override { return router_context_; }
  Envoy::ProcessContextOptRef processContext() override {
    PANIC("NighthawkServerInstance::processContext not implemented");
  }
  Envoy::ThreadLocal::Instance& threadLocal() override { return tls_; }
  Envoy::LocalInfo::LocalInfo& localInfo() const override { return local_info_; }
  Envoy::TimeSource& timeSource() override { return api_.timeSource(); }
  void flushStats() override { PANIC("NighthawkServerInstance::flushStats not implemented"); }
  Envoy::ProtobufMessage::ValidationContext& messageValidationContext() override {
    return validation_context_;
  }
  Envoy::ProtobufMessage::ValidationVisitor& messageValidationVisitor() override {
    return validation_context_.staticValidationVisitor();
  }
  Envoy::Server::Configuration::StatsConfig& statsConfig() override {
    PANIC("NighthawkServerInstance::statsConfig not implemented");
  }
  envoy::config::bootstrap::v3::Bootstrap& bootstrap() override {
    PANIC("NighthawkServerInstance::bootstrap not implemented");
  }
  Envoy::Server::Configuration::ServerFactoryContext& serverFactoryContext() override {
    return server_factory_context_;
  }
  Envoy::Server::Configuration::TransportSocketFactoryContext&
  transportSocketFactoryContext() override {
    PANIC("NighthawkServerInstance::transportSocketFactoryContext not implemented");
  }
  void setDefaultTracingConfig(const envoy::config::trace::v3::Tracing&) override {
    PANIC("NighthawkServerInstance::setDefaultTracingConfig not implemented");
  }
  bool enableReusePortDefault() override {
    PANIC("NighthawkServerInstance::enableReusePortDefault not implemented");
  }
  void setSinkPredicates(std::unique_ptr<Envoy::Stats::SinkPredicates>&&) override {
    PANIC("NighthawkServerInstance::setSinkPredicates not implemented");
  }
  Envoy::Config::XdsManager& xdsManager() override { return xds_manager_; }
  Envoy::Regex::Engine& regexEngine() override {
    PANIC("NighthawkServerInstance::regexEngine not implemented");
  };

private:
  Envoy::OptRef<Envoy::Server::Admin> admin_;
  Envoy::Api::Api& api_;
  Envoy::Event::Dispatcher& dispatcher_;
  Envoy::AccessLog::AccessLogManager& log_manager_;
  Envoy::Server::Options& options_;
  Envoy::Runtime::Loader& runtime_;
  Envoy::Singleton::Manager& singleton_manager_;
  Envoy::Stats::StoreRoot& stats_store_;
  Envoy::ThreadLocal::Instance& tls_;
  Envoy::LocalInfo::LocalInfo& local_info_;
  Envoy::ProtobufMessage::ProdValidationContextImpl& validation_context_;
  Envoy::Grpc::Context& grpc_context_;
  Envoy::Http::Context& http_context_;
  Envoy::Router::Context& router_context_;
  NighthawkServerFactoryContext server_factory_context_;
  Envoy::Http::HttpServerPropertiesCacheManagerImpl http_server_properties_cache_manager_;
  Envoy::Config::XdsManagerImpl xds_manager_;
  NighthawkLifecycleNotifierImpl lifecycle_notifier_; // A no-op object that lives here.
  Envoy::Secret::SecretManagerImpl& secret_manager_;
  std::unique_ptr<Envoy::Server::OverloadManager>
      null_overload_manager_; // Created in the constructor.
};

/**
 * Compiles a list of factories and the configurations they will use to create plugins.
 *
 * @param options The options used for initializing the process
 * @return std::vector<std::pair<TypedExtensionConfig, UserDefinedOutputPluginFactory*>> vector of
 * pairs, each containing a factory and its corresponding configuration.
 */
std::vector<UserDefinedOutputConfigFactoryPair>
getUserDefinedFactoryConfigPairs(const Options& options) {
  std::vector<UserDefinedOutputConfigFactoryPair> factory_config_pairs;
  for (const TypedExtensionConfig& config : options.userDefinedOutputPluginConfigs()) {
    auto* factory = Envoy::Config::Utility::getAndCheckFactory<UserDefinedOutputPluginFactory>(
        config, /*is_optional=*/false);
    UserDefinedOutputConfigFactoryPair pair(config, factory);
    factory_config_pairs.push_back(pair);
  }
  return factory_config_pairs;
}

/**
 * Takes in a single worker's user defined outputs and collects them into the
 * user_defined_outputs_by_plugin map. When called for each worker, transforms the outputs from
 * being mapped by Worker to being mapped by Plugin.
 *
 * @param user_defined_outputs_by_plugin The map that this function will collect the worker data
 * into, maps plugin name to the set of results for that plugin.
 * @param worker_user_defined_outputs The per worker results to collect and organize.
 */
void collectUserDefinedOutputs(
    absl::flat_hash_map<std::string, std::vector<nighthawk::client::UserDefinedOutput>>&
        user_defined_outputs_by_plugin,
    const std::vector<nighthawk::client::UserDefinedOutput>& worker_user_defined_outputs) {
  for (const nighthawk::client::UserDefinedOutput& user_defined_output :
       worker_user_defined_outputs) {
    absl::string_view plugin_name = user_defined_output.plugin_name();
    if (!user_defined_outputs_by_plugin.contains(plugin_name)) {
      user_defined_outputs_by_plugin[plugin_name] = {user_defined_output};
    } else {
      user_defined_outputs_by_plugin[plugin_name].push_back(user_defined_output);
    }
  }
}

/**
 * For each provided user defined output plugin factory, aggregates all of its corresponding results
 * into a global user defined output.
 *
 * @param user_defined_outputs_by_plugin A map of plugin name to the set of collected results for
 * that plugin.
 * @param user_defined_output_factories A vector of the plugin factories used in this execution
 * process.
 * @return std::vector<nighthawk::client::UserDefinedOutput> The aggregated global results for each
 * plugin.
 */
std::vector<nighthawk::client::UserDefinedOutput> compileGlobalUserDefinedPluginOutputs(
    const absl::flat_hash_map<std::string, std::vector<nighthawk::client::UserDefinedOutput>>&
        user_defined_outputs_by_plugin,
    const std::vector<UserDefinedOutputConfigFactoryPair>& user_defined_output_factories) {
  std::vector<nighthawk::client::UserDefinedOutput> global_outputs;
  for (const UserDefinedOutputConfigFactoryPair& config_factory_pair :
       user_defined_output_factories) {
    UserDefinedOutputPluginFactory* factory = config_factory_pair.second;
    nighthawk::client::UserDefinedOutput global_output;
    global_output.set_plugin_name(factory->name());

    auto it = user_defined_outputs_by_plugin.find(factory->name());
    if (it != user_defined_outputs_by_plugin.end()) {
      absl::StatusOr<Envoy::ProtobufWkt::Any> global_output_any =
          factory->AggregateGlobalOutput(it->second);
      if (global_output_any.ok()) {
        *global_output.mutable_typed_output() = *global_output_any;
      } else {
        *global_output.mutable_error_message() = global_output_any.status().ToString();
      }
    } else {
      *global_output.mutable_error_message() =
          "No per worker outputs found for a factory when performing aggregation";
    }
    global_outputs.push_back(global_output);
  }
  return global_outputs;
}

// Disables the hot restart Envoy functionality.
std::string HotRestartDisabled(bool) { return "hot restart is disabled"; }

} // namespace

// We customize ProdClusterManagerFactory for the sole purpose of returning our specialized
// http1 pool to the benchmark client, which allows us to offer connection prefetching.
class ClusterManagerFactory : public Envoy::Upstream::ProdClusterManagerFactory {
public:
  using Envoy::Upstream::ProdClusterManagerFactory::ProdClusterManagerFactory;

  Envoy::Http::ConnectionPool::InstancePtr allocateConnPool(
      Envoy::Event::Dispatcher& dispatcher, Envoy::Upstream::HostConstSharedPtr host,
      Envoy::Upstream::ResourcePriority priority, std::vector<Envoy::Http::Protocol>& protocols,
      const absl::optional<envoy::config::core::v3::AlternateProtocolsCacheOptions>&
          alternate_protocol_options,
      const Envoy::Network::ConnectionSocket::OptionsSharedPtr& options,
      const Envoy::Network::TransportSocketOptionsConstSharedPtr& transport_socket_options,
      Envoy::TimeSource& time_source, Envoy::Upstream::ClusterConnectivityState& state,
      Envoy::Http::PersistentQuicInfoPtr& quic_info,
      Envoy::OptRef<Envoy::Quic::EnvoyQuicNetworkObserverRegistry> network_observer_registry)
      override {
    // This changed in
    // https://github.com/envoyproxy/envoy/commit/93ee668a690d297ab5e8bd2cbf03771d852ebbda ALPN may
    // be set up to negotiate a protocol, in which case we'd need a HttpConnPoolImplMixed. However,
    // our integration tests pass, and for now this might suffice. In case we do run into the need
    // for supporting multiple protocols in a single pool, ensure we hear about it soon, by
    // asserting.
    RELEASE_ASSERT(protocols.size() == 1, "Expected a single protocol in protocols vector.");
    const Envoy::Http::Protocol& protocol = protocols[0];
    if (protocol == Envoy::Http::Protocol::Http11 || protocol == Envoy::Http::Protocol::Http10) {
      auto* h1_pool = new Http1PoolImpl(
          host, priority, dispatcher, options, transport_socket_options,
          context_.api().randomGenerator(), state,
          [](Envoy::Http::HttpConnPoolImplBase* pool) {
            return std::make_unique<Envoy::Http::Http1::ActiveClient>(*pool, absl::nullopt);
          },
          [](Envoy::Upstream::Host::CreateConnectionData& data,
             Envoy::Http::HttpConnPoolImplBase* pool) {
            Envoy::Http::CodecClientPtr codec{new Envoy::Http::CodecClientProd(
                Envoy::Http::CodecClient::Type::HTTP1, std::move(data.connection_),
                data.host_description_, pool->dispatcher(), pool->randomGenerator(),
                pool->transportSocketOptions())};
            return codec;
          },
          protocols, context_.overloadManager());
      h1_pool->setConnectionReuseStrategy(connection_reuse_strategy_);
      h1_pool->setPrefetchConnections(prefetch_connections_);
      return Envoy::Http::ConnectionPool::InstancePtr{h1_pool};
    }
    return Envoy::Upstream::ProdClusterManagerFactory::allocateConnPool(
        dispatcher, host, priority, protocols, alternate_protocol_options, options,
        transport_socket_options, time_source, state, quic_info, network_observer_registry);
  }

  void setConnectionReuseStrategy(
      const Http1PoolImpl::ConnectionReuseStrategy connection_reuse_strategy) {
    connection_reuse_strategy_ = connection_reuse_strategy;
  }
  void setPrefetchConnections(const bool prefetch_connections) {
    prefetch_connections_ = prefetch_connections;
  }

private:
  Http1PoolImpl::ConnectionReuseStrategy connection_reuse_strategy_{};
  bool prefetch_connections_{};
};

ProcessImpl::ProcessImpl(const Options& options, Envoy::Event::TimeSystem& time_system,
                         Envoy::Network::DnsResolverFactory& dns_resolver_factory,
                         TypedExtensionConfig typed_dns_resolver_config,
                         const std::shared_ptr<Envoy::ProcessWide>& process_wide)
    : options_(options), number_of_workers_(BootstrapFactory::determineConcurrency(options_)),
      process_wide_(process_wide == nullptr ? std::make_shared<Envoy::ProcessWide>()
                                            : process_wide),
      time_system_(time_system), stats_allocator_(symbol_table_), store_root_(stats_allocator_),
      quic_stat_names_(store_root_.symbolTable()),
      api_(std::make_unique<Envoy::Api::Impl>(platform_impl_.threadFactory(), store_root_,
                                              time_system_, platform_impl_.fileSystem(), generator_,
                                              bootstrap_)),
      dispatcher_(api_->allocateDispatcher("main_thread")), benchmark_client_factory_(options),
      termination_predicate_factory_(options), sequencer_factory_(options),
      request_generator_factory_(options, *api_), init_manager_("nh_init_manager"),
      local_info_(new Envoy::LocalInfo::LocalInfoImpl(
          store_root_.symbolTable(), node_, node_context_params_,
          Envoy::Network::Utility::getLocalAddress(Envoy::Network::Address::IpVersion::v4),
          "nighthawk_service_zone", "nighthawk_service_cluster", "nighthawk_service_node")),
      secret_manager_(config_tracker_), http_context_(store_root_.symbolTable()),
      grpc_context_(store_root_.symbolTable()),
      singleton_manager_(std::make_unique<Envoy::Singleton::ManagerImpl>()),
      access_log_manager_(std::chrono::milliseconds(1000), *api_, *dispatcher_, access_log_lock_,
                          store_root_),
      dns_resolver_factory_(dns_resolver_factory),
      typed_dns_resolver_config_(std::move(typed_dns_resolver_config)),
      init_watcher_("Nighthawk", []() {}),
      admin_(Envoy::Network::Address::InstanceConstSharedPtr()),
      validation_context_(false, false, false, false), router_context_(store_root_.symbolTable()),
      envoy_options_(/* args = */ {"process_impl"}, HotRestartDisabled, spdlog::level::info) {
  // Any dispatchers created after the following call will use hr timers.
  setupForHRTimers();
  std::string lower = absl::AsciiStrToLower(
      nighthawk::client::Verbosity::VerbosityOptions_Name(options_.verbosity()));
  configureComponentLogLevels(spdlog::level::from_str(lower));
}

absl::StatusOr<ProcessPtr> ProcessImpl::CreateProcessImpl(
    const Options& options, Envoy::Network::DnsResolverFactory& dns_resolver_factory,
    TypedExtensionConfig typed_dns_resolver_config, Envoy::Event::TimeSystem& time_system,
    const std::shared_ptr<Envoy::ProcessWide>& process_wide) {
  std::unique_ptr<ProcessImpl> process(new ProcessImpl(options, time_system, dns_resolver_factory,
                                                       std::move(typed_dns_resolver_config),
                                                       process_wide));

  absl::StatusOr<Bootstrap> bootstrap = createBootstrapConfiguration(
      *process->dispatcher_, *process->api_, process->options_, process->dns_resolver_factory_,
      process->typed_dns_resolver_config_, process->number_of_workers_);
  if (!bootstrap.ok()) {
    ENVOY_LOG(error, "Failed to create bootstrap configuration: {}", bootstrap.status().message());
    process->shutdown();
    return bootstrap.status();
  }

  // Ideally we would create the bootstrap first and then pass it to the
  // constructor of Envoy::Api::Api. That cannot be done because of a circular
  // dependency:
  // 1) The constructor of Envoy::Api::Api requires an instance of Bootstrap.
  // 2) The bootstrap generator requires an Envoy::Event::Dispatcher to resolve
  //    URIs to IPs required in the Bootstrap.
  // 3) The constructor of Envoy::Event::Dispatcher requires Envoy::Api::Api.
  //
  // Replacing the bootstrap_ after the Envoy::Api::Api has been created is
  // assumed to be safe, because we still do it while constructing the
  // ProcessImpl, i.e. before we start running the process.
  process->bootstrap_ = *bootstrap;
  process->user_defined_output_factories_ = getUserDefinedFactoryConfigPairs(options);

  return process;
}

ProcessImpl::~ProcessImpl() {
  RELEASE_ASSERT(shutdown_, "shutdown not called before destruction.");
}

void ProcessImpl::shutdown() {
  // Before we shut down the worker threads, stop threading.
  tls_.shutdownGlobalThreading();
  store_root_.shutdownThreading();

  {
    auto guard = std::make_unique<Envoy::Thread::LockGuard>(workers_lock_);
    // flush_worker_->shutdown() needs to happen before workers_.clear() so that
    // metrics defined in workers scope will be included in the final stats
    // flush which happens in FlushWorkerImpl::shutdownThread() after
    // flush_worker_->shutdown() is called. For the order between worker shutdown() and
    // shutdownThread(), see worker_impl.cc.
    if (flush_worker_) {
      flush_worker_->shutdown();
    }
    // Before shutting down the cluster manager, stop the workers.
    for (auto& worker : workers_) {
      worker->shutdown();
    }
    workers_.clear();
  }
  if (cluster_manager_ != nullptr) {
    cluster_manager_->shutdown();
  }
  tls_.shutdownThread();
  dispatcher_->shutdown();
  shutdown_ = true;
}

bool ProcessImpl::requestExecutionCancellation() {
  ENVOY_LOG(debug, "Requesting workers to cancel execution");
  auto guard = std::make_unique<Envoy::Thread::LockGuard>(workers_lock_);
  for (auto& worker : workers_) {
    worker->requestExecutionCancellation();
  }
  cancelled_ = true;
  return true;
}

Envoy::MonotonicTime
ProcessImpl::computeFirstWorkerStart(Envoy::Event::TimeSystem& time_system,
                                     const absl::optional<Envoy::SystemTime>& scheduled_start,
                                     const uint32_t concurrency) {
  const std::chrono::nanoseconds first_worker_delay =
      scheduled_start.has_value() ? scheduled_start.value() - time_system.systemTime()
                                  : 500ms + (concurrency * 50ms);
  const Envoy::MonotonicTime monotonic_now = time_system.monotonicTime();
  const Envoy::MonotonicTime first_worker_start = monotonic_now + first_worker_delay;
  return first_worker_start;
}

std::chrono::nanoseconds ProcessImpl::computeInterWorkerDelay(const uint32_t concurrency,
                                                              const uint32_t rps) {
  const double inter_worker_delay_usec = (1. / rps) * 1000000 / concurrency;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(inter_worker_delay_usec * 1us);
}

absl::Status ProcessImpl::createWorkers(const uint32_t concurrency,
                                        const absl::optional<Envoy::SystemTime>& scheduled_start) {
  ASSERT(workers_.empty());
  const Envoy::MonotonicTime first_worker_start =
      computeFirstWorkerStart(time_system_, scheduled_start, concurrency);
  const std::chrono::nanoseconds inter_worker_delay =
      computeInterWorkerDelay(concurrency, options_.requestsPerSecond());
  int worker_number = 0;
  while (workers_.size() < concurrency) {
    absl::StatusOr<std::vector<UserDefinedOutputNamePluginPair>> plugins =
        createUserDefinedOutputPlugins(user_defined_output_factories_, worker_number);
    if (!plugins.ok()) {
      return plugins.status();
    }
    workers_.push_back(std::make_unique<ClientWorkerImpl>(
        *api_, tls_, cluster_manager_, benchmark_client_factory_, termination_predicate_factory_,
        sequencer_factory_, request_generator_factory_, store_root_, worker_number,
        first_worker_start + (inter_worker_delay * worker_number), tracer_,
        options_.simpleWarmup() ? ClientWorkerImpl::HardCodedWarmupStyle::ON
                                : ClientWorkerImpl::HardCodedWarmupStyle::OFF,
        std::move(*plugins)));
    worker_number++;
  }
  return absl::OkStatus();
}

void ProcessImpl::configureComponentLogLevels(spdlog::level::level_enum level) {
  // TODO(oschaaf): Add options to tweak the log level of the various log tags
  // that are available.
  Envoy::Logger::Registry::setLogLevel(level);
  Envoy::Logger::Logger* logger_to_change = Envoy::Logger::Registry::logger("main");
  logger_to_change->setLevel(level);
}

std::vector<StatisticPtr>
ProcessImpl::vectorizeStatisticPtrMap(const StatisticPtrMap& statistics) const {
  std::vector<StatisticPtr> v;
  for (const auto& statistic : statistics) {
    // Clone the original statistic into a new one.
    auto new_statistic =
        statistic.second->createNewInstanceOfSameType()->combine(*(statistic.second));
    new_statistic->setId(statistic.first);
    v.push_back(std::move(new_statistic));
  }
  return v;
}

std::vector<StatisticPtr>
ProcessImpl::mergeWorkerStatistics(const std::vector<ClientWorkerPtr>& workers) const {
  // First we init merged_statistics with newly created statistics instances.
  // We do that by adding the same amount of Statistic instances that the first worker has.
  // (We always have at least one worker, and all workers have the same number of Statistic
  // instances associated to them, in the same order).
  std::vector<StatisticPtr> merged_statistics;
  StatisticPtrMap w0_statistics = workers[0]->statistics();
  for (const auto& w0_statistic : w0_statistics) {
    auto new_statistic = w0_statistic.second->createNewInstanceOfSameType();
    new_statistic->setId(w0_statistic.first);
    merged_statistics.push_back(std::move(new_statistic));
  }

  // Merge the statistics of all workers into the statistics vector we initialized above.
  for (auto& w : workers) {
    uint32_t i = 0;
    for (const auto& wx_statistic : w->statistics()) {
      auto merged = merged_statistics[i]->combine(*(wx_statistic.second));
      merged->setId(merged_statistics[i]->id());
      merged_statistics[i] = std::move(merged);
      i++;
    }
  }
  return merged_statistics;
}

void ProcessImpl::addTracingCluster(envoy::config::bootstrap::v3::Bootstrap& bootstrap,
                                    const Uri& uri) const {
  auto* cluster = bootstrap.mutable_static_resources()->add_clusters();
  cluster->set_name("tracing");
  cluster->mutable_connect_timeout()->set_seconds(options_.timeout().count());
  cluster->set_type(
      envoy::config::cluster::v3::Cluster::DiscoveryType::Cluster_DiscoveryType_STATIC);
  auto* load_assignment = cluster->mutable_load_assignment();
  load_assignment->set_cluster_name(cluster->name());
  auto* socket = cluster->mutable_load_assignment()
                     ->add_endpoints()
                     ->add_lb_endpoints()
                     ->mutable_endpoint()
                     ->mutable_address()
                     ->mutable_socket_address();
  socket->set_address(uri.address()->ip()->addressAsString());
  socket->set_port_value(uri.port());
}

void ProcessImpl::setupTracingImplementation(envoy::config::bootstrap::v3::Bootstrap& bootstrap,
                                             const Uri& uri) const {
#ifdef ZIPKIN_ENABLED
  auto* http = bootstrap.mutable_tracing()->mutable_http();
  auto scheme = uri.scheme();
  const std::string kTracingClusterName = "tracing";
  http->set_name(fmt::format("envoy.{}", scheme));
  RELEASE_ASSERT(scheme == "zipkin", "Only zipkin is supported");
  envoy::config::trace::v3::ZipkinConfig config;
  config.mutable_collector_cluster()->assign(kTracingClusterName);
  config.mutable_collector_endpoint()->assign(std::string(uri.path()));
  config.set_collector_endpoint_version(envoy::config::trace::v3::ZipkinConfig::HTTP_JSON);
  config.mutable_shared_span_context()->set_value(true);
  http->mutable_typed_config()->PackFrom(config);
#else
  ENVOY_LOG(error, "Not build with any tracing support");
  UNREFERENCED_PARAMETER(bootstrap);
  UNREFERENCED_PARAMETER(uri);
#endif
}

void ProcessImpl::maybeCreateTracingDriver(const envoy::config::trace::v3::Tracing& configuration) {
  if (configuration.has_http()) {
#ifdef ZIPKIN_ENABLED
    std::string type = configuration.http().name();
    ENVOY_LOG(info, "loading tracing driver: {}", type);
    // Envoy::Server::Configuration::TracerFactory would be useful here to create the right
    // tracer implementation for us. However that ends up needing a Server::Instance to be passed
    // in which we do not have, and creating a fake for that means we risk code-churn because of
    // upstream code changes.
    auto& factory =
        Envoy::Config::Utility::getAndCheckFactory<Envoy::Server::Configuration::TracerFactory>(
            configuration.http());
    Envoy::ProtobufTypes::MessagePtr message = Envoy::Config::Utility::translateToFactoryConfig(
        configuration.http(), Envoy::ProtobufMessage::getStrictValidationVisitor(), factory);
    const auto* zipkin_config =
        Envoy::Protobuf::DynamicCastToGenerated<const envoy::config::trace::v3::ZipkinConfig>(
            message.get());
    Envoy::Tracing::DriverPtr zipkin_driver =
        std::make_unique<Envoy::Extensions::Tracers::Zipkin::Driver>(
            *zipkin_config, *cluster_manager_, scope_root_, tls_, *runtime_loader_.get(),
            *local_info_, generator_, time_system_);
    tracer_ = std::make_unique<Envoy::Tracing::TracerImpl>(std::move(zipkin_driver), *local_info_);
#else
    ENVOY_LOG(error, "Not build with any tracing support");
#endif
  }
}

void ProcessImpl::setupStatsSinks(const envoy::config::bootstrap::v3::Bootstrap& bootstrap,
                                  std::list<std::unique_ptr<Envoy::Stats::Sink>>& stats_sinks) {
  for (const envoy::config::metrics::v3::StatsSink& stats_sink : bootstrap.stats_sinks()) {
    ENVOY_LOG(info, "loading stats sink configuration in Nighthawk");
    auto& factory =
        Envoy::Config::Utility::getAndCheckFactory<NighthawkStatsSinkFactory>(stats_sink);
    stats_sinks.emplace_back(factory.createStatsSink(store_root_.symbolTable()));
  }
  for (std::unique_ptr<Envoy::Stats::Sink>& sink : stats_sinks) {
    store_root_.addSink(*sink);
  }
}

bool ProcessImpl::runInternal(OutputCollector& collector, const UriPtr& tracing_uri,
                              const Envoy::Network::DnsResolverSharedPtr& dns_resolver,
                              const absl::optional<Envoy::SystemTime>& scheduled_start) {
  const Envoy::SystemTime now = time_system_.systemTime();
  if (scheduled_start.value_or(now) < now) {
    ENVOY_LOG(error, "Scheduled execution date already transpired.");
    return false;
  }
  {
    auto guard = std::make_unique<Envoy::Thread::LockGuard>(workers_lock_);
    if (cancelled_) {
      return true;
    }
    shutdown_ = false;

    // Needs to happen as early as possible (before createWorkers()) in the instantiation to preempt
    // the objects that require stats.
    if (!options_.statsSinks().empty()) {
      absl::StatusOr<Envoy::Stats::TagProducerPtr> producer_or_error =
          Envoy::Stats::TagProducerImpl::createTagProducer(bootstrap_.stats_config(),
                                                           envoy_options_.statsTags());
      if (!producer_or_error.ok()) {
        ENVOY_LOG(error, "createTagProducer failed. Received bad status: {}",
                  producer_or_error.status());
        return false;
      }
      store_root_.setTagProducer(std::move(producer_or_error.value()));
    }

    absl::Status workers_status = createWorkers(number_of_workers_, scheduled_start);
    if (!workers_status.ok()) {
      ENVOY_LOG(error, "createWorkers failed. Received bad status: {}", workers_status.message());
      return false;
    }
    tls_.registerThread(*dispatcher_, true);
    store_root_.initializeThreading(*dispatcher_, tls_);

    absl::StatusOr<Envoy::Runtime::LoaderPtr> loader = Envoy::Runtime::LoaderImpl::create(
        *dispatcher_, tls_, {}, *local_info_, store_root_, generator_,
        Envoy::ProtobufMessage::getStrictValidationVisitor(), *api_);

    if (!loader.ok()) {
      ENVOY_LOG(error, "create runtime loader failed. Received bad status: {}", loader.status());
      return false;
    }

    runtime_loader_ = *std::move(loader);

    server_ = std::make_unique<NighthawkServerInstance>(
        admin_, *api_, *dispatcher_, access_log_manager_, envoy_options_, *runtime_loader_.get(),
        *singleton_manager_, tls_, *local_info_, validation_context_, grpc_context_, http_context_,
        router_context_, store_root_, secret_manager_);
    ssl_context_manager_ =
        std::make_unique<Envoy::Extensions::TransportSockets::Tls::ContextManagerImpl>(
            server_->serverFactoryContext());
    dynamic_cast<NighthawkServerFactoryContext*>(&server_->serverFactoryContext())
        ->setSslContextManager(*ssl_context_manager_);
    cluster_manager_factory_ = std::make_unique<ClusterManagerFactory>(
        server_->serverFactoryContext(),
        [dns_resolver]() -> Envoy::Network::DnsResolverSharedPtr { return dns_resolver; },
        quic_stat_names_);
    cluster_manager_factory_->setConnectionReuseStrategy(
        options_.h1ConnectionReuseStrategy() == nighthawk::client::H1ConnectionReuseStrategy::LRU
            ? Http1PoolImpl::ConnectionReuseStrategy::LRU
            : Http1PoolImpl::ConnectionReuseStrategy::MRU);
    cluster_manager_factory_->setPrefetchConnections(options_.prefetchConnections());
    if (tracing_uri != nullptr) {
      setupTracingImplementation(bootstrap_, *tracing_uri);
      addTracingCluster(bootstrap_, *tracing_uri);
    }
    ENVOY_LOG(debug, "Computed configuration: {}", absl::StrCat(bootstrap_));
    absl::StatusOr<Envoy::Upstream::ClusterManagerPtr> cluster_manager =
        cluster_manager_factory_->clusterManagerFromProto(bootstrap_);
    if (!cluster_manager.ok()) {
      ENVOY_LOG(error, "clusterManagerFromProto failed. Received bad status: {}",
                cluster_manager.status().message());
      return false;
    }
    cluster_manager_ = std::move(*cluster_manager);
    dynamic_cast<NighthawkServerFactoryContext*>(&server_->serverFactoryContext())
        ->setClusterManager(*cluster_manager_);
    absl::Status status = cluster_manager_->initialize(bootstrap_);
    if (!status.ok()) {
      ENVOY_LOG(error, "cluster_manager initialize failed. Received bad status: {}",
                status.message());
      return false;
    }
    maybeCreateTracingDriver(bootstrap_.tracing());
    cluster_manager_->setInitializedCb(
        [this]() -> void { init_manager_.initialize(init_watcher_); });

    absl::Status initialize_status = runtime_loader_->initialize(*cluster_manager_);
    if (!initialize_status.ok()) {
      ENVOY_LOG(error, "runtime_loader initialize failed. Received bad status: {}",
                initialize_status.message());
      return false;
    }

    std::list<std::unique_ptr<Envoy::Stats::Sink>> stats_sinks;
    setupStatsSinks(bootstrap_, stats_sinks);
    std::chrono::milliseconds stats_flush_interval = std::chrono::milliseconds(
        Envoy::DurationUtil::durationToMilliseconds(bootstrap_.stats_flush_interval()));

    if (!options_.statsSinks().empty()) {
      // There should be only a single live flush worker instance at any time.
      flush_worker_ = std::make_unique<FlushWorkerImpl>(
          stats_flush_interval, *api_, tls_, store_root_, stats_sinks, *cluster_manager_);
      flush_worker_->start();
    }

    for (auto& w : workers_) {
      w->start();
    }
  }
  for (auto& w : workers_) {
    w->waitForCompletion();
  }

  if (!options_.statsSinks().empty() && flush_worker_ != nullptr) {
    // Stop the running dispatcher in flush_worker_. Needs to be called after all
    // client workers are complete so that all the metrics can be flushed.
    flush_worker_->exitDispatcher();
    flush_worker_->waitForCompletion();
  }

  int i = 0;
  std::chrono::nanoseconds total_execution_duration = 0ns;
  absl::optional<Envoy::SystemTime> first_acquisition_time = absl::nullopt;
  // Maps registered user defined output plugin name to the output results for every worker's plugin
  // of that name.
  absl::flat_hash_map<std::string, std::vector<nighthawk::client::UserDefinedOutput>>
      user_defined_outputs_by_plugin{};
  for (auto& worker : workers_) {
    auto sequencer_execution_duration = worker->phase().sequencer().executionDuration();
    absl::optional<Envoy::SystemTime> worker_first_acquisition_time =
        worker->phase().sequencer().rate_limiter().firstAcquisitionTime();
    if (worker_first_acquisition_time.has_value()) {
      first_acquisition_time =
          first_acquisition_time.has_value()
              ? std::min(first_acquisition_time.value(), worker_first_acquisition_time.value())
              : worker_first_acquisition_time.value();
    }
    std::vector<nighthawk::client::UserDefinedOutput> worker_user_defined_outputs =
        worker->getUserDefinedOutputResults();
    collectUserDefinedOutputs(user_defined_outputs_by_plugin, worker_user_defined_outputs);
    // We don't write per-worker results if we only have a single worker, because the global
    // results will be precisely the same.
    if (workers_.size() > 1) {
      StatisticFactoryImpl statistic_factory(options_);
      collector.addResult(fmt::format("worker_{}", i),
                          vectorizeStatisticPtrMap(worker->statistics()),
                          worker->threadLocalCounterValues(), sequencer_execution_duration,
                          worker_first_acquisition_time, worker_user_defined_outputs);
    }
    total_execution_duration += sequencer_execution_duration;
    i++;
  }

  // Note that above we use use counter values snapshotted by the workers right after its
  // execution completes. Here we query the live counters to get to the global numbers. To make
  // sure the global aggregated numbers line up, we must take care not to shut down the benchmark
  // client before we do this, as that will increment certain counters like connections closed,
  // etc.
  const std::map<std::string, uint64_t>& counters = Utility().mapCountersFromStore(
      store_root_, [](absl::string_view, uint64_t value) { return value > 0; });
  StatisticFactoryImpl statistic_factory(options_);
  std::vector<nighthawk::client::UserDefinedOutput> global_user_defined_outputs =
      compileGlobalUserDefinedPluginOutputs(user_defined_outputs_by_plugin,
                                            user_defined_output_factories_);
  collector.addResult("global", mergeWorkerStatistics(workers_), counters,
                      total_execution_duration / workers_.size(), first_acquisition_time,
                      global_user_defined_outputs);
  if (counters.find("sequencer.failed_terminations") == counters.end()) {
    return true;
  } else {
    ENVOY_LOG(error, "Terminated early because of a failure predicate.");
    ENVOY_LOG(
        info,
        "Check the output for problematic counter values. The default Nighthawk failure predicates "
        "report failure if (1) Nighthawk could not connect to the target (see "
        "'benchmark.pool_connection_failure' counter; check the address and port number, and try "
        "explicitly setting --address-family v4 or v6, especially when using DNS; instead of "
        "localhost try 127.0.0.1 or ::1 explicitly), (2) the protocol was not supported by the "
        "target (see 'benchmark.stream_resets' counter; check http/https in the URI, --h2), (3) "
        "the target returned a 4xx or 5xx HTTP response code (see 'benchmark.http_4xx' and "
        "'benchmark.http_5xx' counters; check the URI path and the server config), or (4) a custom "
        "gRPC RequestSource failed. --failure-predicate can be used to relax expectations.");
    return false;
  }
}

bool ProcessImpl::run(OutputCollector& collector) {
  UriPtr tracing_uri;

  absl::StatusOr<Envoy::Network::DnsResolverSharedPtr> dns_resolver =
      dns_resolver_factory_.createDnsResolver(*dispatcher_, *api_, typed_dns_resolver_config_);
  if (!dns_resolver.ok()) {
    ENVOY_LOG(error, "Failed to create DNS resolver: {}", dns_resolver.status());
    return false;
  }
  try {
    if (options_.trace() != "") {
      tracing_uri = std::make_unique<UriImpl>(options_.trace());
      tracing_uri->resolve(*dispatcher_, *dns_resolver.value(),
                           Utility::translateFamilyOptionString(options_.addressFamily()));
    }
  } catch (const UriException& ex) {
    ENVOY_LOG(error,
              "URI exception (for example, malformed URI syntax, bad "
              "MultiTarget path, unresolvable host DNS): {}",
              ex.what());
    return false;
  }

  try {
    return runInternal(collector, tracing_uri, *dns_resolver, options_.scheduled_start());
  } catch (Envoy::EnvoyException& ex) {
    ENVOY_LOG(error, "Fatal EnvoyException exception: {}", ex.what());
    throw;
  } catch (NighthawkException& ex) {
    ENVOY_LOG(error, "Fatal NighthawkException exception: {}", ex.what());
    throw;
  }
}

void ProcessImpl::setupForHRTimers() {
  // We override the local environment to indicate to libevent that we favor precision over
  // efficiency. Note that it is also possible to do this at setup time via libevent's api's.
  // The upside of the approach below is that we are very loosely coupled and have a one-liner.
  // Getting to libevent for the other approach is going to introduce more code as we would need to
  // derive our own customized versions of certain Envoy concepts.
  putenv(const_cast<char*>("EVENT_PRECISE_TIMER=1"));
}

} // namespace Client
} // namespace Nighthawk
