// frontend/src/app/login/page.tsx
"use client";
import React, { useState, useEffect, useRef } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../../lib/firebase";
import { Loader2, BrainCircuit, ShieldCheck, Lock } from "lucide-react";

// --- AUTHORIZATION WHITELIST ---
const ALLOWED_EMAILS = [
  "1MEHK23@manipal.in",
  "2AIMM22@aiims.in"
];

// --- 1. THE INTERACTIVE BRAIN NODE CANVAS ---
const InteractiveBrainNodes = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let particles: Particle[] = [];
    const particleCount = 280; 
    const connectionDistance = 50; 
    const mouseDistance = 250; 

    let mouse = { x: -1000, y: -1000 };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    const handleMouseLeave = () => {
      mouse.x = -1000;
      mouse.y = -1000;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseleave", handleMouseLeave);

    const isInsideBrain = (nx: number, ny: number) => {
      const cerebrum = Math.pow(nx / 1.0, 2) + Math.pow((ny + 0.1) / 0.75, 2) <= 1;
      const cerebellum = Math.pow(nx - 0.45, 2) + Math.pow(ny - 0.45, 2) <= 0.12;
      const stem = nx > 0.15 && nx < 0.35 && ny > 0.4 && ny < 0.95;
      const faceCutout = nx < -0.3 && ny > 0.15;
      return (cerebrum || cerebellum || stem) && !faceCutout;
    };

    class Particle {
      anchorX: number;
      anchorY: number;
      x: number;
      y: number;
      size: number;
      angle: number;
      speed: number;
      radius: number;

      constructor(width: number, height: number, scale: number) {
        let valid = false;
        let nx = 0, ny = 0;
        
        while (!valid) {
          nx = (Math.random() - 0.5) * 2.5; 
          ny = (Math.random() - 0.5) * 2.5;
          valid = isInsideBrain(nx, ny);
        }

        this.anchorX = width / 2 + nx * scale;
        this.anchorY = height / 2 + ny * scale;
        this.x = this.anchorX;
        this.y = this.anchorY;
        
        this.size = Math.random() * 2.5 + 1.2; 
        this.angle = Math.random() * Math.PI * 2;
        this.speed = Math.random() * 0.02 + 0.005;
        this.radius = Math.random() * 10 + 2; 
      }

      update(mouse: {x: number, y: number}) {
        this.angle += this.speed;
        let targetX = this.anchorX + Math.cos(this.angle) * this.radius;
        let targetY = this.anchorY + Math.sin(this.angle) * this.radius;

        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < mouseDistance) {
          const force = (mouseDistance - distance) / mouseDistance;
          targetX -= (dx / distance) * force * 45; 
          targetY -= (dy / distance) * force * 45;
        }

        this.x += (targetX - this.x) * 0.1;
        this.y += (targetY - this.y) * 0.1;
      }

      draw(ctx: CanvasRenderingContext2D) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 240, 255, 0.9)";
        ctx.fill();
      }
    }

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      const brainScale = Math.min(canvas.width, canvas.height) * 0.35;

      particles = [];
      for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle(canvas.width, canvas.height, brainScale));
      }
    };

    window.addEventListener("resize", resize);
    resize();

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < particles.length; i++) {
        particles[i].update(mouse);
        particles[i].draw(ctx);

        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < connectionDistance) {
            ctx.beginPath();
            ctx.strokeStyle = `rgba(0, 240, 255, ${0.4 - (distance / connectionDistance) * 0.4})`;
            ctx.lineWidth = 0.8;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }
      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 z-0 pointer-events-auto"
      style={{ background: "transparent" }}
    />
  );
};


// --- 2. MAIN COMPONENT ---
export default function LandingLoginPage() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    const cleanEmail = email.trim();

    if (!ALLOWED_EMAILS.includes(cleanEmail)) {
      setError("Unauthorized credentials. Institutional access required.");
      setIsLoading(false);
      return;
    }

    try {
      await signInWithEmailAndPassword(auth, cleanEmail, password);
      window.location.href = "/";
    } catch (err: any) {
      setError("Invalid authorization key or unregistered node.");
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#030305] text-white font-sans relative overflow-hidden flex flex-col">
      
      {/* Custom levitation animation */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-15px); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
      `}} />

      {/* The Interactive Neural Brain */}
      <InteractiveBrainNodes />
      
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[50vw] h-[50vw] bg-[#00f0ff]/5 blur-[120px] rounded-full pointer-events-none z-0" />

      {/* Top Center Logo & Branding */}
      <div className="absolute top-10 left-1/2 -translate-x-1/2 flex items-center space-x-4 z-20 pointer-events-auto">
        <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center backdrop-blur-md shadow-[0_0_20px_rgba(0,240,255,0.2)]">
          {/* Logo precisely scaled to match 32pt text */}
          <BrainCircuit className="h-10 w-10 text-[#00f0ff]" />
        </div>
        {/* Exactly 32pt Text */}
        <span className="text-[32px] font-extrabold tracking-wider drop-shadow-lg">
          NeuroScan<span className="text-[#00f0ff]">AI</span>
        </span>
      </div>

      {/* Hero Content */}
      <main className="flex-1 flex flex-col items-center justify-center z-10 pointer-events-none mt-20">
        <h1 className="text-6xl md:text-8xl font-extrabold tracking-tighter text-center leading-[1.1] text-transparent bg-clip-text bg-gradient-to-b from-white to-white/60 drop-shadow-2xl">
          Adversarially Robust <br /> Segmentation
        </h1>
        <p className="mt-6 text-gray-400 max-w-2xl text-center text-lg md:text-xl leading-relaxed font-light">
          Authorized clinical personnel only. Encrypted ARMT-GAN neural framework.
        </p>

        {/* The Levitating Glass Button - Disappears when modal is open */}
        {!isModalOpen && (
          <div className="mt-16 pointer-events-auto animate-float">
            <button 
              onClick={() => setIsModalOpen(true)}
              className="group relative overflow-hidden rounded-full bg-white/5 backdrop-blur-2xl border border-white/20 px-14 py-6 shadow-[0_0_40px_rgba(0,240,255,0.2)] hover:shadow-[0_0_60px_rgba(0,240,255,0.5)] hover:bg-white/10 transition-all duration-500 flex items-center space-x-4"
            >
              <ShieldCheck className="h-7 w-7 text-[#00f0ff] relative z-10" />
              <span className="relative z-10 text-xl font-bold tracking-wide text-white">Initialize Secure Session</span>
              
              <div className="absolute inset-0 bg-gradient-to-r from-[#00f0ff]/10 to-[#7000ff]/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            </button>
          </div>
        )}
      </main>

      {/* Centered Pop-up Login Vault Overlay */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-md px-4 animate-in fade-in duration-300">
          
          {/* Pure Glass Effect Modal Box */}
          <div className="w-full max-w-md p-10 rounded-[2rem] bg-white/[0.03] backdrop-blur-3xl border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.5)] relative transform transition-all scale-100 flex flex-col items-center">
            
            {/* Close Button */}
            <button 
              onClick={() => setIsModalOpen(false)}
              className="absolute top-6 right-6 h-8 w-8 rounded-full bg-white/5 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
            >
              ✕
            </button>
            
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#00f0ff]/20 to-[#7000ff]/20 border border-white/10 flex items-center justify-center mb-6 shadow-[0_0_20px_rgba(0,240,255,0.2)]">
              <Lock className="h-8 w-8 text-[#00f0ff]" />
            </div>

            {/* Exactly 26pt Matching Heading */}
            <h2 className="text-[26px] font-extrabold text-white mb-2 tracking-tight">LOGIN</h2>
            <p className="text-sm text-gray-400 mb-8 font-medium">Medical Platform</p>

            <form onSubmit={handleLogin} className="space-y-5 w-full">
              <div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Institutional ID (Email)"
                  className="w-full bg-white/[0.05] border border-white/10 rounded-2xl py-4 px-5 text-white placeholder-gray-500 focus:outline-none focus:border-[#00f0ff]/50 focus:bg-white/10 transition-all text-sm font-medium shadow-inner"
                />
              </div>

              <div>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Authorization Key"
                  className="w-full bg-white/[0.05] border border-white/10 rounded-2xl py-4 px-5 text-white placeholder-gray-500 focus:outline-none focus:border-[#00f0ff]/50 focus:bg-white/10 transition-all text-sm font-medium shadow-inner"
                />
              </div>

              {error && (
                <div className="text-red-400 text-xs bg-red-500/10 p-4 rounded-xl border border-red-500/20 font-medium flex items-start">
                  <span className="mr-2 mt-0.5">⚠️</span>
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-4 mt-4 rounded-2xl bg-gradient-to-r from-[#00f0ff] to-[#7000ff] text-black font-bold tracking-widest text-sm hover:opacity-90 hover:scale-[1.02] transition-all duration-300 disabled:opacity-50 flex items-center justify-center space-x-2 shadow-[0_0_20px_rgba(0,240,255,0.4)]"
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin text-black" />
                ) : (
                  <span>AUTHENTICATE</span>
                )}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}