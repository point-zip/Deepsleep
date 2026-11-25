/** 接口协议：决定请求体、流式事件格式与鉴权方式 */
export type ProtocolId = 'anthropic' | 'openai' | 'responses' | 'gemini'

/**
 * 协议能力，界面据此禁用或隐藏不支持的开关：
 * - thinking：是否支持思考内容（OpenAI 兼容的 reasoning_content、Gemini 的 thought 等）
 * - search：是否支持服务端联网搜索
 * - title：是否由服务端生成会话标题，否则首轮提问后本地截取
 * - files：附件传递方式，api 为上传后按文件 ID 引用，inline 为随请求内联图片，none 为不支持
 */
export type ProtocolFeatures = {
  thinking: boolean
  search: boolean
  title: boolean
  files: 'api' | 'inline' | 'none'
}

/** 供应商的服务端扩展；只有带扩展的供应商才发送对应的私有字段 */
export type ProviderExtensions = {
  /** DeepSeek 扩展：deepseek_web_search / deepseek_read_url 工具、safety_check、服务端标题与 latest_reminder 角色 */
  deepseek: boolean
}

/** 内置供应商模板；新建供应商时复制为可编辑的配置 */
export type ProviderPreset = {
  id: string
  name: string
  /** 控制台地址，用于「获取 API Key」 */
  url: string
  protocol: ProtocolId
  baseUrl: string
  /** 常用模型；留空表示需要手动填写或从接口拉取 */
  models: string[]
  /** 本地部署的服务可以不带 API Key */
  keyOptional?: boolean
  /** 在协议默认能力之上覆盖的能力 */
  features?: Partial<ProtocolFeatures>
  extensions?: Partial<ProviderExtensions>
}

/** 用户保存的供应商配置；API Key 与选中的模型按供应商分别保存 */
export type ProviderConfig = {
  id: string
  /** 来源模板，决定默认能力和展示信息；自定义供应商为 custom */
  presetId: string
  name: string
  protocol: ProtocolId
  baseUrl: string
  apiKey: string
  /** 可选模型列表，可手动编辑或从接口拉取 */
  models: string[]
  /** 当前选中的模型 */
  model: string
}

/**
 * 工具服务：搜索、抓取、OCR、ASR 与 TTS 共用的接口地址与 Key。
 * 这些能力由官方 DeepSeek 以独立 REST 接口提供，与对话供应商分开配置，
 * 换用其他对话模型时仍然可以使用。
 */
export type ToolsConfig = {
  baseUrl: string
  apiKey: string
}

/**
 * 协议默认能力；供应商模板可以覆盖其中几项。
 * search 表示协议自带服务端联网搜索：Anthropic 声明 web_search，
 * Responses 声明 web_search 内置工具，Gemini 声明 googleSearch；
 * OpenAI Chat Completions 没有标准内置搜索，只能靠客户端工具服务补齐。
 */
export const PROTOCOL_FEATURES: Record<ProtocolId, ProtocolFeatures> = {
  anthropic: { thinking: true, search: true, title: false, files: 'api' },
  openai: { thinking: true, search: false, title: false, files: 'inline' },
  responses: { thinking: true, search: true, title: false, files: 'inline' },
  gemini: { thinking: true, search: true, title: false, files: 'inline' },
}

/** 协议展示名，设置页的协议下拉框使用 */
export const PROTOCOL_LABELS: Record<ProtocolId, string> = {
  anthropic: 'Anthropic Messages',
  openai: 'OpenAI Chat Completions',
  responses: 'OpenAI Responses',
  gemini: 'Google Gemini',
}

export const PROTOCOL_IDS: ProtocolId[] = ['anthropic', 'openai', 'responses', 'gemini']

export const isProtocolId = (value: unknown): value is ProtocolId =>
  typeof value === 'string' && (PROTOCOL_IDS as string[]).includes(value)
