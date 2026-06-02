import "./globals.css";

export const metadata = {
  title: "Harry Potter Films Q&A",
  description: "Ask anything about the Harry Potter films.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
