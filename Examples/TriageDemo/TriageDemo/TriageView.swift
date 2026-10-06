import SwiftUI

/// The one screen: a message, three sample messages, and Verdict's three answers as bars.
struct TriageView: View {
    let loader: ModelLoader
    @Bindable var model: TriageModel
    @FocusState private var editing: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // While typing, the keyboard covers the lower half of the screen; the intro
                    // and the samples step aside so the answers stay in view under the message.
                    if !editing {
                        Text("Three typed questions about a customer message, answered by Verdict.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    editor
                    if !editing {
                        samples
                    }
                    status
                    if let note = model.note {
                        Label(note, systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    if let error = model.error {
                        Label(error, systemImage: "xmark.octagon")
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                    if loader.phase == .ready {
                        onDevice
                    }
                    ForEach(model.rows) { row in
                        AnswerCard(row: row)
                    }
                }
                .padding()
            }
            .scrollDismissesKeyboard(.interactively)
            .animation(.smooth(duration: 0.25), value: editing)
            .navigationTitle("Triage")
            .navigationBarTitleDisplayMode(editing ? .inline : .large)
            .toolbar {
                ToolbarItem(placement: .keyboard) {
                    HStack {
                        Spacer()
                        Button("Done") { editing = false }
                    }
                }
            }
        }
        // Keyed on the text and the engine: a keystroke cancels the pending read and starts a
        // new debounced one; the engine's arrival answers whatever was typed while it loaded.
        .task(id: TaskKey(text: model.text, ready: model.engine != nil)) {
            await model.answer()
        }
    }

    private struct TaskKey: Equatable {
        var text: String
        var ready: Bool
    }

    private var editor: some View {
        TextField("Type or paste a customer message", text: $model.text, axis: .vertical)
            .lineLimit(3...8)
            .focused($editing)
            .padding(12)
            .background(.background.secondary, in: .rect(cornerRadius: 12))
            .accessibilityLabel("Customer message")
    }

    private var samples: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Try a sample")
                .font(.footnote.weight(.semibold))
                .foregroundStyle(.secondary)
            ViewThatFits(in: .horizontal) {
                HStack { sampleButtons }
                VStack(alignment: .leading) { sampleButtons }
            }
        }
    }

    @ViewBuilder private var sampleButtons: some View {
        ForEach(TriageQuestions.samples) { sample in
            Button(sample.name) {
                model.text = sample.text
                editing = false
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.roundedRectangle(radius: 8))
            .accessibilityHint("Puts the sample message in the text field")
        }
    }

    @ViewBuilder private var status: some View {
        switch loader.phase {
        case .idle, .ready:
            EmptyView()
        case .downloading(let received, let total):
            VStack(alignment: .leading, spacing: 6) {
                Text("Downloading Verdict, once")
                    .font(.headline)
                ProgressView(value: Double(received), total: Double(max(total, 1)))
                Text(
                    "\(received.formatted(.byteCount(style: .file))) of "
                        + "\(total.formatted(.byteCount(style: .file))). Each file's SHA-256 is "
                        + "checked; later launches work offline."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .combine)
        case .loading:
            HStack(spacing: 10) {
                ProgressView()
                Text("Loading Verdict")
                    .font(.subheadline)
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 10) {
                Label(message, systemImage: "wifi.exclamationmark")
                    .font(.subheadline)
                Button("Try again") {
                    Task { await loader.load() }
                }
                .buttonStyle(.borderedProminent)
                .buttonBorderShape(.roundedRectangle(radius: 8))
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.12), in: .rect(cornerRadius: 12))
        }
    }

    /// Where the answers come from, and how long the last read took in the model.
    private var onDevice: some View {
        HStack(spacing: 6) {
            Image(systemName: "iphone")
            Text("On this device")
            if let time = model.modelTime {
                Text(verbatim: "·")
                Text(
                    "\(Int((Double(time.components.attoseconds) / 1e15).rounded()) + Int(time.components.seconds) * 1000) ms"
                )
            }
            Spacer()
            Text(verbatim: "verdict-1.4")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("on-device")
    }
}

/// One question's answer: the question, the answer, the confidence and a bar per outcome.
struct AnswerCard: View {
    let row: TriageModel.Row
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(row.title)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                // At accessibility sizes the confidence goes under the answer rather than beside it.
                let layout =
                    typeSize.isAccessibilitySize
                    ? AnyLayout(VStackLayout(alignment: .leading, spacing: 2))
                    : AnyLayout(HStackLayout(alignment: .firstTextBaseline))
                layout {
                    Text(row.headline)
                        .font(.title2.weight(.semibold))
                        .contentTransition(.opacity)
                    if !typeSize.isAccessibilitySize {
                        Spacer()
                    }
                    Text(
                        "confidence \(row.confidence.formatted(.percent.precision(.fractionLength(0))))"
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("answer-\(row.id)")
            ForEach(row.bars) { bar in
                ProbabilityBar(
                    bar: bar,
                    highlighted: bar.label
                        == row.bars.max { $0.probability < $1.probability }?.label
                )
                .accessibilityIdentifier("bar-\(row.id)-\(bar.label)")
            }
        }
        .padding(14)
        .background(.background.secondary, in: .rect(cornerRadius: 14))
        .animation(.smooth(duration: 0.35), value: row)
    }
}

/// An outcome's label, its probability as a bar, and the percentage.
struct ProbabilityBar: View {
    let bar: TriageModel.Bar
    let highlighted: Bool
    @ScaledMetric(relativeTo: .body) private var barHeight = 10
    @ScaledMetric(relativeTo: .body) private var labelWidth = 72
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        let layout =
            typeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 4))
            : AnyLayout(HStackLayout(spacing: 10))
        layout {
            Text(bar.label)
                .font(.subheadline)
                .frame(width: typeSize.isAccessibilitySize ? nil : labelWidth, alignment: .leading)
            HStack(spacing: 10) {
                GeometryReader { proxy in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(.quaternary)
                        RoundedRectangle(cornerRadius: 3)
                            .fill(highlighted ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                            .frame(width: proxy.size.width * min(max(bar.probability, 0), 1))
                    }
                }
                .frame(height: barHeight)
                Text(bar.probability.formatted(.percent.precision(.fractionLength(0))))
                    .font(.subheadline.monospacedDigit())
                    .frame(minWidth: 44, alignment: .trailing)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(bar.label)
        .accessibilityValue(bar.probability.formatted(.percent.precision(.fractionLength(0))))
    }
}

#Preview {
    TriageView(loader: ModelLoader(), model: TriageModel())
}
