//
//  DevicePreviewFrame.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Lightweight "device-shaped" wrapper for SwiftUI Previews. Adds a
//  rounded frame, a faux status bar, and a faux home indicator so
//  Previews look like a real iPhone without pulling in the actual
//  simulator bezel.
//
//  Use only in `#Preview` blocks. Never in production views — those
//  should render to the full screen.

import SwiftUI

#if DEBUG

public struct DevicePreviewFrame<Content: View>: View {

    @Environment(\.themeTokens) private var tokens

    public let content: Content
    public let width: CGFloat
    public let dark: Bool

    public init(width: CGFloat = 390, dark: Bool = false, @ViewBuilder content: () -> Content) {
        self.width = width
        self.dark = dark
        self.content = content()
    }

    public var body: some View {
        let height = width * (844.0 / 390.0)
        let cornerRadius = width * 0.115

        ZStack(alignment: .top) {
            content
                .frame(width: width, height: height)
            statusBar
            VStack { Spacer(); homeIndicator }
                .frame(width: width, height: height)
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .shadow(color: .black.opacity(0.18), radius: 22, x: 0, y: 14)
        .overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(Color.black.opacity(0.1), lineWidth: 1)
        )
    }

    private var statusBar: some View {
        HStack {
            Text("9:41")
                .font(.system(size: 13, weight: .semibold))
            Spacer()
            Capsule()
                .stroke(lineWidth: 1)
                .frame(width: 22, height: 11)
                .overlay(
                    Capsule()
                        .padding(2)
                )
        }
        .foregroundStyle(dark ? .white : .black)
        .padding(.horizontal, 30)
        .padding(.top, 14)
        .frame(width: width, height: 54)
    }

    private var homeIndicator: some View {
        Capsule()
            .fill(dark ? Color.white.opacity(0.7) : Color.black.opacity(0.3))
            .frame(width: 100, height: 4)
            .padding(.bottom, 8)
    }
}

#endif
