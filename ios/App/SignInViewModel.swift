import Foundation
import PickleballUpload

/// Drives `SignInView`: a minimal email/password + invite-code flow over
/// `AuthApiClient` (INFRA-4). Registration always chains straight into login
/// so a successful "create account" also leaves the user signed in.
@MainActor
final class SignInViewModel: ObservableObject {
    enum Mode: Equatable {
        case signIn
        case createAccount

        var title: String {
            switch self {
            case .signIn: return "Sign in"
            case .createAccount: return "Create account"
            }
        }

        var toggleLabel: String {
            switch self {
            case .signIn: return "Need an account? Create one"
            case .createAccount: return "Already have an account? Sign in"
            }
        }
    }

    @Published private(set) var mode: Mode = .signIn
    @Published var email: String = ""
    @Published var password: String = ""
    @Published var inviteCode: String = ""
    @Published private(set) var isSubmitting = false
    @Published private(set) var errorMessage: String?

    private let authApiClient: AuthApiClient
    private let onSignedIn: () -> Void

    init(authApiClient: AuthApiClient = AuthApiClient(), onSignedIn: @escaping () -> Void) {
        self.authApiClient = authApiClient
        self.onSignedIn = onSignedIn
    }

    var canSubmit: Bool {
        guard !isSubmitting else { return false }
        guard email.contains("@"), password.count >= 8 else { return false }
        if mode == .createAccount {
            return !inviteCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return true
    }

    func switchMode(to newMode: Mode) {
        guard newMode != mode else { return }
        mode = newMode
        errorMessage = nil
    }

    func submit() async {
        guard canSubmit else { return }
        isSubmitting = true
        errorMessage = nil
        defer { isSubmitting = false }

        do {
            if mode == .createAccount {
                _ = try await authApiClient.register(email: email, password: password, inviteCode: inviteCode)
            }
            _ = try await authApiClient.login(email: email, password: password)
            onSignedIn()
        } catch let AuthApiClientError.httpStatus(status, detail) {
            errorMessage = Self.readableMessage(status: status, detail: detail)
        } catch {
            errorMessage = "Could not reach the server. Check your connection and try again."
        }
    }

    private static func readableMessage(status: Int, detail: String) -> String {
        switch status {
        case 401:
            return "Incorrect email or password."
        case 403:
            return "That invite code isn't valid."
        case 409:
            return "An account with that email already exists."
        case 422:
            return detail.isEmpty ? "Check your email and password." : detail
        default:
            return detail.isEmpty ? "Something went wrong (\(status))." : detail
        }
    }
}
