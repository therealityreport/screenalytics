import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "SCREENALYTICS",
  description: "Reality TV face tracking and screen time analytics platform",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main>{children}</main>
      </body>
    </html>
  );
}
