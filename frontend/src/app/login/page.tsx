"use client";
import React, { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { useRouter } from "next/navigation";
import { Loader2, BrainCircuit, Lock } from "lucide-react";
import { BrainNodes } from "@/components/brain-nodes";

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
        <div className="min-h-screen bg-[#030305] text-white font-sans relative overflow-hidden flex items-center justify-center p-4">
            <BrainNodes />
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[50vw] h-[50vw] bg-neuro-cyan/5 blur-[120px] rounded-full pointer-events-none z-0" />

            <div className="absolute top-8 left-1/2 -translate-x-1/2 flex items-center space-x-3 z-40">
                <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center backdrop-blur-md">
                    <BrainCircuit className="h-5 w-5 text-neuro-cyan" />
                </div>
                <span className="text-xl font-semibold tracking-wide text-white">
                    NeuroScan<span className="text-neuro-cyan">AI</span>
                </span>
            </div>

            <div className="w-full max-w-md p-8 sm:p-10 rounded-[2rem] bg-card/80 backdrop-blur-3xl border border-white/10 shadow-[0_20px_60px_rgba(0,0,0,0.6)] relative z-10 flex flex-col items-center animate-in fade-in zoom-in-95 duration-500">
                <div className="w-14 h-14 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 shadow-[0_0_25px_rgba(56,224,255,0.2)]">
                    <Lock className="h-6 w-6 text-neuro-cyan" />
                </div>

                <h2 className="text-2xl font-semibold text-white mb-2 tracking-tight">Institutional Access</h2>
                <p className="text-xs text-white/50 mb-8 text-center font-medium">Establish a verified clinical session for ARMT-GAN inference.</p>

                <form onSubmit={handleLogin} className="space-y-4 w-full">
                    <div>
                        <input
                            type="email"
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Institutional ID (Email)"
                            className="w-full bg-white/[0.05] border border-white/10 rounded-2xl py-3.5 px-4 text-white placeholder-white/30 focus:outline-none focus:border-neuro-cyan/60 text-sm font-medium transition-colors"
                        />
                    </div>

                    <div>
                        <input
                            type="password"
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Authorization Key / Password"
                            className="w-full bg-white/[0.05] border border-white/10 rounded-2xl py-3.5 px-4 text-white placeholder-white/30 focus:outline-none focus:border-neuro-cyan/60 text-sm font-medium transition-colors"
                        />
                    </div>

                    {error && (
                        <div className="text-red-400 text-xs bg-red-500/10 p-3.5 rounded-xl border border-red-500/20 font-medium">
                            ⚠️ {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full py-3.5 mt-2 rounded-2xl bg-white text-black font-semibold tracking-wide text-sm hover:bg-white/90 transition-all flex items-center justify-center space-x-2 shadow-[0_0_20px_rgba(255,255,255,0.2)] disabled:opacity-50 cursor-pointer"
                    >
                        {isLoading ? <Loader2 className="h-4 w-4 animate-spin text-black" /> : <span>AUTHENTICATE</span>}
                    </button>
                </form>
            </div>
        </div>
    );
}