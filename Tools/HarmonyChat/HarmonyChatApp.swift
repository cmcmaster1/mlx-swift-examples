import Foundation
import MLXLLM
@preconcurrency import MLXLMCommon
import Tokenizers

@main
struct HarmonyChatApp {

    static func main() async {
        do {
            try await run()
        } catch {
            fputs("Error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run() async throws {
        let options = try CLIOptions.parse()
        prepareMetalEnvironment()
        print("🛠️  Loading mlx-community/gpt-oss-20b-MXFP4-Q8 ...")

        let configuration = LLMRegistry.gpt_oss_20b_MXFP4_Q8
        let container = try await LLMModelFactory.shared.loadContainer(configuration: configuration)

        var history: [TextMessage] = []
        if !options.systemPrompt.isEmpty {
            history.append(.init(role: Chat.Message.Role.system.rawValue, content: options.systemPrompt))
        }
        if !options.developerPrompt.isEmpty {
            history.append(.init(role: Chat.Message.Role.developer.rawValue, content: options.developerPrompt))
        }

        if let reasoning = options.reasoningEffort {
            print("➡️  Reasoning effort: \(reasoning.rawValue)")
        } else {
            print("➡️  Reasoning effort: default (medium)")
        }
        print("✅ Model ready. Type '/exit' to quit.\n")

        while true {
            guard let userLine = readInput(prompt: "user> ") else {
                break
            }

            let trimmed = userLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            if trimmed == "/exit" { break }

            history.append(.init(role: Chat.Message.Role.user.rawValue, content: trimmed))

            let (payloadJSON, promptPreview, rawResponse) = try await container.perform(values: history) { context, snapshot -> (String, String, String) in
                let chatMessages = snapshot.map { item -> Chat.Message in
                    let role = Chat.Message.Role(rawValue: item.role) ?? .user
                    return Chat.Message(role: role, content: item.content)
                }

                var userInput = UserInput(chat: chatMessages)
                var additional = userInput.additionalContext ?? [:]
                if let reasoning = options.reasoningEffort {
                    additional["reasoning_effort"] = reasoning.rawValue
                }
                userInput.additionalContext = additional.isEmpty ? nil : additional

                let payload = HarmonyMessageGenerator().generate(messages: chatMessages)
                let promptTokens = try context.tokenizer.applyChatTemplate(
                    messages: payload,
                    chatTemplate: .literal(HarmonyMessageGenerator.chatTemplate),
                    addGenerationPrompt: true,
                    truncation: false,
                    maxLength: nil,
                    tools: userInput.tools,
                    additionalContext: userInput.additionalContext
                )
                let promptString = context.tokenizer.decode(tokens: promptTokens)

                var parameters = GenerateParameters()
                parameters.reasoningEffort = options.reasoningEffort

                let lmInput = try await context.processor.prepare(input: userInput)
                let result = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: parameters,
                    context: context
                ) { (_: [Int]) -> GenerateDisposition in
                    .more
                }

                let payloadJSON: String
                if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]),
                    let text = String(data: data, encoding: .utf8)
                {
                    payloadJSON = text
                } else {
                    payloadJSON = "[]"
                }

                return (payloadJSON, promptString, result.output)
            }

            print("--- Harmony Messages ---")
            print(payloadJSON)
            print("------------------------\n")

            print("\n--- Harmony Prompt (truncated) ---")
            print(promptPreview.prefix(400))
            print("----------------------------------\n")

            let segments = HarmonyResponseParser.parse(rawResponse)
            if options.showAnalysis {
                for segment in segments where segment.channel == "analysis" {
                    print(
                        "[analysis] \(segment.content.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines))\n"
                    )
                }
            }

            guard let finalSegment = segments.last(where: { $0.channel == "final" }) else {
                let fallback = rawResponse.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                print("assistant> \(fallback)\n")
                history.append(.init(role: Chat.Message.Role.assistant.rawValue, content: fallback))
                continue
            }

            let reply = finalSegment.content.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
            print("assistant> \(reply)\n")
            history.append(.init(role: Chat.Message.Role.assistant.rawValue, content: reply))
        }

        print("👋 Bye!")
    }

    private static func readInput(prompt: String) -> String? {
        fputs(prompt, stdout)
        return readLine()
    }

    private static func prepareMetalEnvironment() {
        let env = ProcessInfo.processInfo.environment
        if env["MLX_METAL_PATH"] != nil {
            return
        }

        let fm = FileManager.default
        var candidates: [URL] = []

        if let executableURL = Bundle.main.executableURL {
            let buildDir = executableURL.deletingLastPathComponent()
            candidates.append(buildDir.appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources"))
        }

        let derivedDataRoot = fm.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Developer/Xcode/DerivedData", isDirectory: true)
        if fm.fileExists(atPath: derivedDataRoot.path) {
            if let enumerator = fm.enumerator(
                at: derivedDataRoot,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles, .skipsPackageDescendants])
            {
                for case let url as URL in enumerator {
                    if url.lastPathComponent == "mlx-swift_Cmlx.bundle" {
                        let resources = url.appendingPathComponent("Contents/Resources")
                        candidates.append(resources)
                        break
                    }
                }
            }
        }

        for url in candidates {
            let metallib = url.appendingPathComponent("default.metallib")
            if fm.fileExists(atPath: metallib.path) {
                setenv("MLX_METAL_PATH", url.path, 1)
                print("ℹ️  Using metallib at \(url.path)")
                return
            }
        }

        print("⚠️  Warning: Unable to locate default.metallib automatically. Set MLX_METAL_PATH manually if MLX fails to load.")
    }
}

private struct TextMessage: Sendable {
    var role: String
    var content: String
}

private struct CLIOptions: Sendable {
    var systemPrompt: String
    var developerPrompt: String
    var reasoningEffort: GenerateParameters.ReasoningEffort?
    var showAnalysis: Bool

    static func parse() throws -> CLIOptions {
        var systemPrompt = "You are ChatGPT, a large language model trained by OpenAI."
        var developerPrompt = "Answer questions helpfully and keep responses concise."
        var reasoning: GenerateParameters.ReasoningEffort? = .medium
        var showAnalysis = false

        var args = Array(CommandLine.arguments.dropFirst())
        while let arg = args.first {
            args.removeFirst()
            switch arg {
            case "--system":
                guard let value = args.first else { throw CLIError.missingValue("--system") }
                systemPrompt = value
                args.removeFirst()
            case "--developer":
                guard let value = args.first else { throw CLIError.missingValue("--developer") }
                developerPrompt = value
                args.removeFirst()
            case "--reasoning":
                guard let value = args.first else { throw CLIError.missingValue("--reasoning") }
                reasoning = GenerateParameters.ReasoningEffort(argument: value)
                args.removeFirst()
            case "--analysis":
                showAnalysis = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                throw CLIError.unknownArgument(arg)
            }
        }

        return CLIOptions(
            systemPrompt: systemPrompt,
            developerPrompt: developerPrompt,
            reasoningEffort: reasoning,
            showAnalysis: showAnalysis
        )
    }

    private static func printUsage() {
        let message = """
        harmony-chat usage:
          harmony-chat [--system \"SYSTEM\"] [--developer \"DEVELOPER\"] [--reasoning low|medium|high] [--analysis]

        Options:
          --system       Override the system prompt (default: OpenAI identity string)
          --developer    Developer instructions injected into the Harmony developer message
          --reasoning    Set reasoning effort (low, medium, or high)
          --analysis     Show analysis channel output when present
          --help         Show this help message
        """
        print(message)
    }
}

private enum CLIError: Error, CustomStringConvertible {
    case missingValue(String)
    case unknownArgument(String)

    var description: String {
        switch self {
        case .missingValue(let flag):
            return "Missing value for \(flag)"
        case .unknownArgument(let flag):
            return "Unknown argument: \(flag)"
        }
    }
}

private extension GenerateParameters.ReasoningEffort {
    init?(argument: String) {
        switch argument.lowercased() {
        case "low":
            self = .low
        case "medium":
            self = .medium
        case "high":
            self = .high
        default:
            print("Warning: unsupported reasoning level '\(argument)'. Using default.")
            return nil
        }
    }
}

private struct HarmonySegment {
    let role: String
    let channel: String?
    let recipient: String?
    let content: String

    init(header rawHeader: String, content rawContent: String) {
        let header = rawHeader.trimmingCharacters(in: .whitespacesAndNewlines)
        var role = header
        var channel: String? = nil
        var recipient: String? = nil

        if let channelRange = header.range(of: "<|channel|>") {
            role = String(header[..<channelRange.lowerBound])
            var remainder = String(header[channelRange.upperBound...])

            if let constrainRange = remainder.range(of: "<|constrain|>") {
                remainder = String(remainder[..<constrainRange.lowerBound])
            }

            if let spaceIndex = remainder.firstIndex(of: " ") {
                channel = String(remainder[..<spaceIndex])
                let tail = remainder[spaceIndex...].trimmingCharacters(in: .whitespaces)
                if tail.hasPrefix("to=") {
                    recipient = String(tail.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                }
            } else {
                channel = remainder
            }
        }

        self.role = role.trimmingCharacters(in: .whitespacesAndNewlines)
        self.channel = channel?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.recipient = recipient
        self.content = rawContent
    }
}

private enum HarmonyResponseParser {
    static func parse(_ text: String) -> [HarmonySegment] {
        var segments: [HarmonySegment] = []
        var searchStart = text.startIndex

        while let startRange = text.range(of: "<|start|>", range: searchStart..<text.endIndex) {
            let roleStart = startRange.upperBound
            guard let messageMarker = text.range(of: "<|message|>", range: roleStart..<text.endIndex)
            else { break }

            let header = String(text[roleStart..<messageMarker.lowerBound])
            let contentStart = messageMarker.upperBound

            let endRange = text.range(of: "<|end|>", range: contentStart..<text.endIndex)
                ?? text.range(of: "<|return|>", range: contentStart..<text.endIndex)

            guard let closing = endRange else { break }
            let content = String(text[contentStart..<closing.lowerBound])
            segments.append(HarmonySegment(header: header, content: content))

            searchStart = closing.upperBound
        }

        return segments
    }
}
