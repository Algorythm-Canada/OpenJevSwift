import SwiftUI

/// Triage: three typed questions about a customer message, answered by Verdict on the device.
@main
struct TriageDemoApp: App {
    @State private var loader = ModelLoader()
    @State private var model = TriageModel()

    var body: some Scene {
        WindowGroup {
            TriageView(loader: loader, model: model)
                .task {
                    // `-TriageText "<message>"` at launch fills in the message, for screenshots
                    // and checks that cannot type (UserDefaults reads launch arguments).
                    if let text = UserDefaults.standard.string(forKey: "TriageText") {
                        model.text = text
                    }
                    await loader.load()
                }
                .onChange(of: loader.phase) {
                    if loader.phase == .ready {
                        model.engine = loader.engine
                    }
                }
        }
    }
}
