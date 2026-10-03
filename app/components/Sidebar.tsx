"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, BarChart3, History, Database, Settings } from "lucide-react";
import Image from "next/image";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

interface NavItem {
  label: string;
  href: string;
  icon: React.ElementType;
}

const NAV_ITEMS: NavItem[] = [
  { label: "Home", href: "/", icon: Home },
  { label: "Analytics", href: "/analytics", icon: BarChart3 },
  { label: "History", href: "/history", icon: History },
  { label: "Dataset", href: "/dataset", icon: Database }
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:flex md:flex-col w-(--sidebar-width,240px) shrink-0 bg-sidebar border-r border-border">
      <div className="flex items-center gap-2 px-5 h-16 border-b border-border">
        <Image
          src="/logo.png"
          width={28}
          height={28}
          alt="Lyric-Audio Predictor logo"
        />
        <span className="font-semibold text-sm text-primary">
          Lyric-Audio Predictor
        </span>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ label, href, icon: Icon }) => {
          const active = pathname === href;

          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-primary"
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
