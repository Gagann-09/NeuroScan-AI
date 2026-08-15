"use client"

import { useRef, useEffect } from "react"

// Interactive brain-shaped particle field rendered on a canvas.
// Particles are seeded inside a brain silhouette and repel from the cursor.
export function BrainNodes() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationFrameId: number
    let particles: Particle[] = []
    const particleCount = 260
    const connectionDistance = 52
    const mouseDistance = 220
    const mouse = { x: -1000, y: -1000 }

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX
      mouse.y = e.clientY
    }
    const handleMouseLeave = () => {
      mouse.x = -1000
      mouse.y = -1000
    }

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseleave", handleMouseLeave)

    const isInsideBrain = (nx: number, ny: number) => {
      const cerebrum = Math.pow(nx / 1.0, 2) + Math.pow((ny + 0.1) / 0.75, 2) <= 1
      const cerebellum = Math.pow(nx - 0.45, 2) + Math.pow(ny - 0.45, 2) <= 0.12
      const stem = nx > 0.15 && nx < 0.35 && ny > 0.4 && ny < 0.95
      const faceCutout = nx < -0.3 && ny > 0.15
      return (cerebrum || cerebellum || stem) && !faceCutout
    }

    class Particle {
      anchorX: number
      anchorY: number
      x: number
      y: number
      size: number
      angle: number
      speed: number
      radius: number

      constructor(width: number, height: number, scale: number) {
        let valid = false
        let nx = 0
        let ny = 0
        while (!valid) {
          nx = (Math.random() - 0.5) * 2.5
          ny = (Math.random() - 0.5) * 2.5
          valid = isInsideBrain(nx, ny)
        }
        this.anchorX = width / 2 + nx * scale
        this.anchorY = height / 2 + ny * scale
        this.x = this.anchorX
        this.y = this.anchorY
        this.size = Math.random() * 2.4 + 1.1
        this.angle = Math.random() * Math.PI * 2
        this.speed = Math.random() * 0.02 + 0.005
        this.radius = Math.random() * 10 + 2
      }

      update(m: { x: number; y: number }) {
        this.angle += this.speed
        let targetX = this.anchorX + Math.cos(this.angle) * this.radius
        let targetY = this.anchorY + Math.sin(this.angle) * this.radius
        const dx = m.x - this.x
        const dy = m.y - this.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance < mouseDistance) {
          const force = (mouseDistance - distance) / mouseDistance
          targetX -= (dx / distance) * force * 42
          targetY -= (dy / distance) * force * 42
        }
        this.x += (targetX - this.x) * 0.1
        this.y += (targetY - this.y) * 0.1
      }

      draw(c: CanvasRenderingContext2D) {
        c.beginPath()
        c.arc(this.x, this.y, this.size, 0, Math.PI * 2)
        c.fillStyle = "rgba(56, 224, 255, 0.42)"
        c.fill()
      }
    }

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      const brainScale = Math.min(canvas.width, canvas.height) * 0.35
      particles = []
      for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle(canvas.width, canvas.height, brainScale))
      }
    }

    window.addEventListener("resize", resize)
    resize()

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      for (let i = 0; i < particles.length; i++) {
        particles[i].update(mouse)
        particles[i].draw(ctx)
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const distance = Math.sqrt(dx * dx + dy * dy)
          if (distance < connectionDistance) {
            ctx.beginPath()
            ctx.strokeStyle = `rgba(56, 224, 255, ${0.16 - (distance / connectionDistance) * 0.16})`
            ctx.lineWidth = 0.8
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.stroke()
          }
        }
      }
      animationFrameId = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      window.removeEventListener("resize", resize)
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseleave", handleMouseLeave)
      cancelAnimationFrame(animationFrameId)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="absolute inset-0 z-0"
      style={{ background: "transparent" }}
    />
  )
}
