"use client";
import React, { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { useRouter } from "next/navigation";
import { Loader2, BrainCircuit, ShieldCheck, Lock } from "lucide-react";

const ALLOWED_EMAILS = [
    "1MEHK23@manipal.in",
    "2AIMM22@aiims.in"
];

export default function LoginPage() {
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setIsLoading(true);

        const cleanEmail = email.trim();

        if (!ALLOWED_EMAILS.includes(cleanEmail)) {
            setError("Unauthorized institutional credentials.");
            setIsLoading(false);
            return;
        }

        try {
            await signInWithEmailAndPassword(auth, cleanEmail, password);
            router.push("/");
        } catch (err: any) {
            setError("Invalid authorization key or password.");
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#030305] text-white font-sans relative overflow-hidden flex items-center justify-center">
            <div className="absolute top-10 left-1/2 -translate-x-1/2 flex items-center space-x-4 z-40">
                <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center backdrop-blur-md">
                    <BrainCircuit className="h-8 w-8 text-[#00f0ff]" />
                </div>
                <span className="text-[28px] font-extrabold tracking-wider">
                    NeuroScan<span className="text-[#00f0ff]">AI</span>
                </span>
            </div>

            <div className="w-full max-w-md p-10 rounded-[2rem] bg-white/[0.03] backdrop-blur-3xl border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.5)] relative flex flex-col items-center">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#00f0ff]/20 to-[#7000ff]/20 border border-white/10 flex items-center justify-center mb-6">
                    <Lock className="h-8 w-8 text-[#00f0ff]" />
                </div>

                <h2 className="text-[26px] font-extrabold text-white mb-2 tracking-tight">Node Access</h2>
                <p className="text-sm text-gray-400 mb-8 font-medium">Establish a secure cryptographic handshake.</p>

                <form onSubmit={handleLogin} className="space-y-5 w-full">
                    <input
                        type="email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Institutional ID (Email)"
                        className="w-full bg-white/[0.05] border border-white/10 rounded-2xl py-4 px-5 text-white placeholder-gray-500 focus:outline-none focus:border-[#00f0ff]/50 text-sm font-medium"
                    />

                    <input
                        type="password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Authorization Key"
                        className="w-full bg-white/[0.05] border border-white/10 rounded-2xl py-4 px-5 text-white placeholder-gray-500 focus:outline-none focus:border-[#00f0ff]/50 text-sm font-medium"
                    />

                    {error && (
                        <div className="text-red-400 text-xs bg-red-500/10 p-4 rounded-xl border border-red-500/20 font-medium">
                            ⚠️ {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full py-4 mt-4 rounded-2xl bg-gradient-to-r from-[#00f0ff] to-[#7000ff] text-black font-bold tracking-widest text-sm hover:opacity-90 transition-all flex items-center justify-center space-x-2 shadow-[0_0_20px_rgba(0,240,255,0.4)]"
                    >
                        {isLoading ? <Loader2 className="h-5 w-5 animate-spin text-black" /> : <span>AUTHENTICATE</span>}
                    </button>
                </form>
            </div>
        </div>
    );
}