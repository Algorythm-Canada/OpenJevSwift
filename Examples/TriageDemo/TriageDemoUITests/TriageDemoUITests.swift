import XCTest

#if canImport(XCUIAutomation)
    import XCUIAutomation
#endif

/// Drives the screen as a person would. Verdict must be on the device or downloadable: the first
/// launch fetches about 310 MB, which the timeout allows for.
final class TriageDemoUITests: XCTestCase {
    /// Long enough for a first launch to download, compile and warm up Verdict.
    static let modelTimeout: TimeInterval = 600

    override func setUp() {
        continueAfterFailure = false
    }

    /// A sample gives three answers, each bar labelled for VoiceOver with its outcome and a
    /// percentage.
    @MainActor
    func testSampleShowsThreeAnswers() {
        let app = XCUIApplication()
        app.launch()
        app.buttons["Outage report"].tap()

        let team = app.descendants(matching: .any)["answer-team"]
        XCTAssertTrue(team.waitForExistence(timeout: Self.modelTimeout))
        XCTAssertTrue(team.label.contains("outage"), team.label)
        XCTAssertTrue(app.descendants(matching: .any)["answer-urgent"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["answer-tone"].exists)

        let bars = [
            ("urgent", ["yes", "no"]), ("team", ["outage", "billing", "feature"]),
            ("tone", ["calm", "annoyed", "furious"]),
        ]
        for (question, labels) in bars {
            for label in labels {
                let bar = app.descendants(matching: .any)["bar-\(question)-\(label)"]
                XCTAssertTrue(bar.waitForExistence(timeout: 5), "\(question) \(label)")
                XCTAssertEqual(bar.label, label)
                let value = bar.value as? String ?? ""
                XCTAssertNotNil(value.wholeMatch(of: /\d{1,3}\s?%/), "\(label): \(value)")
            }
        }
        let footer = app.descendants(matching: .any)["on-device"]
        XCTAssertTrue(footer.exists)
        XCTAssertTrue(footer.label.contains("On this device"), footer.label)
    }

    /// Types a billing complaint a few words at a time, so the answers move as the message
    /// grows. The README's recording is this test.
    @MainActor
    func testTypingAMessage() {
        let app = XCUIApplication()
        app.launch()
        let field =
            app.textFields["Customer message"].firstMatch.exists
            ? app.textFields["Customer message"].firstMatch
            : app.textViews["Customer message"].firstMatch
        XCTAssertTrue(field.waitForExistence(timeout: 10))
        // The "On this device" line appears once Verdict is loaded and warmed up.
        XCTAssertTrue(
            app.descendants(matching: .any)["on-device"].waitForExistence(
                timeout: Self.modelTimeout))
        field.tap()

        let words = [
            "Hi,", "you", "charged", "my", "card", "twice", "this", "month", "and", "nobody",
            "answered", "my", "emails.", "Refund", "me", "today", "or", "I", "cancel.",
        ]
        for word in words {
            field.typeText(word + " ")
            Thread.sleep(forTimeInterval: 0.55)
        }
        let team = app.descendants(matching: .any)["answer-team"]
        let billing = NSPredicate(format: "label CONTAINS 'billing'")
        expectation(for: billing, evaluatedWith: team)
        waitForExpectations(timeout: 10)
        Thread.sleep(forTimeInterval: 2)
    }
}
