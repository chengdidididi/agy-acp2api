package plugin

import (
	"encoding/json"
	"net/http"
	"net/url"
)

const (
	ABIVersion    uint32 = 1
	SchemaVersion uint32 = 6 // Version 6+ preserves raw JSON bodies in ManagementResponse without HTML entity escaping

	PluginID   = "agy-acp2api"
	PluginName = "agy-acp2api"

	MethodPluginRegister    = "plugin.register"
	MethodPluginReconfigure = "plugin.reconfigure"
	MethodPluginShutdown    = "plugin.shutdown"

	MethodModelRegister = "model.register"
	MethodModelStatic   = "model.static"
	MethodModelForAuth  = "model.for_auth"
	MethodModelRoute    = "model.route"

	MethodExecutorIdentifier    = "executor.identifier"
	MethodExecutorExecute       = "executor.execute"
	MethodExecutorExecuteStream = "executor.execute_stream"
	MethodExecutorCountTokens   = "executor.count_tokens"
	MethodExecutorHTTPRequest   = "executor.http_request"

	MethodManagementRegister = "management.register"
	MethodManagementHandle   = "management.handle"

	MethodHostStreamEmit  = "host.stream.emit"
	MethodHostStreamClose = "host.stream.close"
)

var Version = "0.1.1"

type Envelope struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *EnvelopeError  `json:"error,omitempty"`
}

type EnvelopeError struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	Retryable  bool   `json:"retryable,omitempty"`
	HTTPStatus int    `json:"http_status,omitempty"`
}

type LifecycleRequest struct {
	ConfigYAML []byte `json:"config_yaml"`
	PluginDir  string `json:"plugin_dir,omitempty"`
}

type Registration struct {
	SchemaVersion uint32       `json:"schema_version"`
	Metadata      Metadata     `json:"metadata"`
	Capabilities  Capabilities `json:"capabilities"`
}

type Metadata struct {
	Name             string        `json:"Name"`
	Version          string        `json:"Version"`
	Author           string        `json:"Author"`
	GitHubRepository string        `json:"GitHubRepository"`
	Logo             string        `json:"Logo,omitempty"`
	ConfigFields     []ConfigField `json:"ConfigFields"`
}

type ConfigField struct {
	Name        string   `json:"Name"`
	Type        string   `json:"Type"`
	Default     any      `json:"Default,omitempty"`
	EnumValues  []string `json:"EnumValues,omitempty"`
	Description string   `json:"Description"`
}

type Capabilities struct {
	ModelRegistrar        bool     `json:"model_registrar,omitempty"`
	ModelProvider         bool     `json:"model_provider,omitempty"`
	ModelRouter           bool     `json:"model_router,omitempty"`
	Executor              bool     `json:"executor,omitempty"`
	ExecutorModelScope    string   `json:"executor_model_scope,omitempty"`
	ExecutorInputFormats  []string `json:"executor_input_formats,omitempty"`
	ExecutorOutputFormats []string `json:"executor_output_formats,omitempty"`
	ManagementAPI         bool     `json:"management_api"`
}

type IdentifierResponse struct {
	Identifier string `json:"identifier"`
}

type ModelInfo struct {
	ID                         string   `json:"ID"`
	Object                     string   `json:"Object,omitempty"`
	Created                    int64    `json:"Created,omitempty"`
	OwnedBy                    string   `json:"OwnedBy,omitempty"`
	Type                       string   `json:"Type,omitempty"`
	DisplayName                string   `json:"DisplayName,omitempty"`
	Name                       string   `json:"Name,omitempty"`
	Description                string   `json:"Description,omitempty"`
	SupportedGenerationMethods []string `json:"SupportedGenerationMethods,omitempty"`
	ContextLength              int64    `json:"ContextLength,omitempty"`
	MaxCompletionTokens        int64    `json:"MaxCompletionTokens,omitempty"`
	UserDefined                bool     `json:"UserDefined,omitempty"`
}

type ModelResponse struct {
	Provider string      `json:"Provider"`
	Models   []ModelInfo `json:"Models"`
}

type ModelRouteRequest struct {
	SourceFormat   string      `json:"SourceFormat"`
	RequestedModel string      `json:"RequestedModel"`
	Stream         bool        `json:"Stream"`
	Headers        http.Header `json:"Headers"`
	Query          url.Values  `json:"Query"`
	Body           []byte      `json:"Body"`
}

type ModelRouteResponse struct {
	Handled     bool   `json:"Handled"`
	TargetKind  string `json:"TargetKind,omitempty"`
	Target      string `json:"Target,omitempty"`
	TargetModel string `json:"TargetModel,omitempty"`
	Reason      string `json:"Reason,omitempty"`
}

type ExecutorRequest struct {
	Model           string      `json:"Model"`
	Format          string      `json:"Format"`
	Stream          bool        `json:"Stream"`
	Headers         http.Header `json:"Headers"`
	Query           url.Values  `json:"Query"`
	OriginalRequest []byte      `json:"OriginalRequest"`
	Payload         []byte      `json:"Payload"`
	StreamID        string      `json:"stream_id,omitempty"`
}

type ExecutorResponse struct {
	Payload []byte      `json:"Payload"`
	Headers http.Header `json:"Headers,omitempty"`
}

type ExecutorStreamChunk struct {
	Payload []byte `json:"Payload,omitempty"`
}

type ExecutorStreamResponse struct {
	Headers http.Header           `json:"headers,omitempty"`
	Chunks  []ExecutorStreamChunk `json:"chunks,omitempty"`
}

type HostStreamEmitRequest struct {
	StreamID string `json:"stream_id"`
	Payload  []byte `json:"payload,omitempty"`
}

type HostStreamCloseRequest struct {
	StreamID string `json:"stream_id"`
	Error    string `json:"error,omitempty"`
}

type ManagementRegistrationRequest struct {
	BasePath         string `json:"BasePath"`
	ResourceBasePath string `json:"ResourceBasePath"`
}

type ManagementRegistrationResponse struct {
	Routes    []ManagementRoute `json:"Routes"`
	Resources []ResourceRoute   `json:"Resources,omitempty"`
}

type ManagementRoute struct {
	Method      string `json:"Method"`
	Path        string `json:"Path"`
	Menu        string `json:"Menu,omitempty"`
	Description string `json:"Description,omitempty"`
}

type ResourceRoute struct {
	Path        string `json:"Path"`
	Menu        string `json:"Menu,omitempty"`
	Description string `json:"Description,omitempty"`
}

type ManagementRequest struct {
	Method  string      `json:"Method"`
	Path    string      `json:"Path"`
	Headers http.Header `json:"Headers"`
	Query   url.Values  `json:"Query"`
	Body    []byte      `json:"Body"`
}

type ManagementResponse struct {
	StatusCode int         `json:"StatusCode,omitempty"`
	Headers    http.Header `json:"Headers,omitempty"`
	Body       []byte      `json:"Body,omitempty"`
}

func OKEnvelope(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return json.Marshal(Envelope{OK: true, Result: raw})
}

func ErrorEnvelope(code, message string, status int) []byte {
	raw, _ := json.Marshal(Envelope{
		OK: false,
		Error: &EnvelopeError{
			Code:       code,
			Message:    message,
			HTTPStatus: status,
		},
	})
	return raw
}
