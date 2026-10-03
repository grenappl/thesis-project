"use client";

import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";
import Image from "next/image";

const PAGE_TITLES: Record<string, string> = {
  "/": "Home",
  "/analytics": "Analytics",
  "/history": "History",
  "/dataset": "Dataset",
};

export default function Header() {
  const pathname = usePathname();
  const title = PAGE_TITLES[pathname] ?? "Lyric-Audio Predictor";

  return (
    <header className="h-16 border-b border-border bg-sidebar flex items-center justify-between px-6">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent text-accent-foreground md:hidden">
          <Image
            src="/logo.png"
            width={28}
            height={28}
            alt="Lyric-Audio Predictor logo"
          />
        </span>
        <span className="font-semibold text-sm text-primary">{title}</span>
      </div>
      <div>
        <ThemeToggle />
      </div>
    </header>
  );
}
