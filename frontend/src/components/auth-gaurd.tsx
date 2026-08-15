"use client";
import { useAuth } from "@/lib/auth-context";
import { useRouter, usePathname } from "next/navigation";
import { useEffect } from "react";
import { Loader2 } from "lucide-react";

export function AuthGuard({ children }: { children: React.ReactNode }) {
    const { user, loading } = useAuth();
    const router = useRouter();
    const pathname = usePathname();

    useEffect(() => {
        if (!loading && !user && pathname !== "/login") {
            router.push("/login");
        }
    }, [user, loading, router, pathname]);

    if (loading) {
        return (
            <div className="min-h-screen bg-[#030305] flex items-center justify-center text-white">
                <Loader2 className="h-8 w-8 animate-spin text-[#00f0ff]" />
            </div>
        );
    }

    if (!user && pathname !== "/login") {
        return null;
    }

    return <>{children}</>;
}