import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "Youth League Team Builder",
  description: "Admin tools for youth league events and teams",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header
          style={{
            padding: "16px 24px",
            borderBottom: "1px solid #e2e8f0",
            background: "white",
          }}
        >
          <strong>Youth League Team Builder</strong>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
