import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "Screenalytics",
  description: "Face tracking and screen time analytics for reality TV",
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
