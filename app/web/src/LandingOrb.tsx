import { useEffect, useRef, useState } from 'react';
import { MODE_DRAWS, resolvePreset } from 'thinking-orbs';

function orbSize() {
  return Math.round(Math.min(window.innerHeight * 0.68, window.innerWidth * 0.86, 720));
}

/** The Connecting orb drawn straight onto a canvas, because the packaged component only ships small sizes. */
export default function LandingOrb() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState(orbSize);

  useEffect(() => {
    const onResize = () => setSize(orbSize());
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * scale;
    canvas.height = size * scale;
    const { mode, opts, speed } = resolvePreset('connecting', 64);
    const draw = (seconds: number) => {
      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.clearRect(0, 0, size, size);
      MODE_DRAWS[mode](context, size, seconds * speed, true, opts);
    };
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      draw(6);
      return;
    }
    let frame = 0;
    const tick = (now: number) => {
      draw(now / 1000);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [size]);

  return <canvas ref={canvasRef} role="img" aria-label="Animated orb" style={{ width: size, height: size }} />;
}
