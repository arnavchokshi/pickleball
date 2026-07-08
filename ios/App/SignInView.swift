import SwiftUI
import PickleballUpload

/// Minimal sign-in / create-account gate (INFRA-4). Deliberately plain --
/// this is plumbing to prove the account flow works end to end, not a
/// branded surface; the DinkVision visual pass is a follow-up.
struct SignInView: View {
    @StateObject private var viewModel: SignInViewModel

    init(authApiClient: AuthApiClient = AuthApiClient(), onSignedIn: @escaping () -> Void) {
        _viewModel = StateObject(wrappedValue: SignInViewModel(authApiClient: authApiClient, onSignedIn: onSignedIn))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(viewModel.mode.title)
                    .font(.system(size: 26, weight: .heavy, design: .rounded))

                TextField("Email", text: $viewModel.email)
                    .textContentType(.emailAddress)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("SignInEmailField")

                SecureField("Password", text: $viewModel.password)
                    .textContentType(viewModel.mode == .signIn ? .password : .newPassword)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityIdentifier("SignInPasswordField")

                if viewModel.mode == .createAccount {
                    TextField("Invite code", text: $viewModel.inviteCode)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityIdentifier("SignInInviteCodeField")
                }

                if let errorMessage = viewModel.errorMessage {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .accessibilityIdentifier("SignInErrorMessage")
                }

                Button {
                    Task { await viewModel.submit() }
                } label: {
                    HStack {
                        Spacer()
                        if viewModel.isSubmitting {
                            ProgressView()
                        } else {
                            Text(viewModel.mode.title)
                        }
                        Spacer()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canSubmit)
                .accessibilityIdentifier("SignInSubmitButton")

                Button {
                    viewModel.switchMode(to: viewModel.mode == .signIn ? .createAccount : .signIn)
                } label: {
                    Text(viewModel.mode.toggleLabel)
                        .font(.footnote)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("SignInModeToggle")
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityIdentifier("DinkVisionScreen-SignIn")
    }
}
