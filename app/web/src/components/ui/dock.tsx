import { AnimatePresence, motion, useMotionValue, useSpring, useTransform, type MotionValue, type SpringOptions } from 'framer-motion';
import { Children, cloneElement, createContext, useContext, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactElement, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

const DOCK_HEIGHT = 128;
const REST_SIZE = 40;
const DEFAULT_MAGNIFICATION = 64;
const DEFAULT_DISTANCE = 120;
const DEFAULT_PANEL_HEIGHT = 60;

type DockContextType = { mouseX: MotionValue<number>; spring: SpringOptions; magnification: number; distance: number };
type DockProps = { children: ReactNode; className?: string; distance?: number; panelHeight?: number; magnification?: number; spring?: SpringOptions };
type DockItemProps = { children: ReactNode; className?: string; label: string; onSelect?: () => void; active?: boolean; disabled?: boolean };
type DockPartProps = { children: ReactNode; className?: string };

const DockContext = createContext<DockContextType | undefined>(undefined);

function useDock() {
  const context = useContext(DockContext);
  if (!context) throw new Error('useDock must be used within a Dock');
  return context;
}

function Dock({ children, className, spring = { mass: 0.1, stiffness: 150, damping: 12 }, magnification = DEFAULT_MAGNIFICATION, distance = DEFAULT_DISTANCE, panelHeight = DEFAULT_PANEL_HEIGHT }: DockProps) {
  const mouseX = useMotionValue(Infinity);
  const isHovered = useMotionValue(0);
  const maxHeight = useMemo(() => Math.max(DOCK_HEIGHT, magnification + magnification / 2 + 4), [magnification]);
  const heightRow = useTransform(isHovered, [0, 1], [panelHeight, maxHeight]);
  const height = useSpring(heightRow, spring);

  return (
    <motion.div style={{ height, scrollbarWidth: 'none' }} className="mx-2 flex max-w-full items-end overflow-x-auto">
      <motion.div
        onMouseMove={({ pageX }) => {
          isHovered.set(1);
          mouseX.set(pageX);
        }}
        onMouseLeave={() => {
          isHovered.set(0);
          mouseX.set(Infinity);
        }}
        className={cn('pointer-events-auto mx-auto flex w-fit gap-3 rounded-2xl px-3', className)}
        style={{ height: panelHeight }}
        role="toolbar"
        aria-label="Navigation"
      >
        <DockContext.Provider value={{ mouseX, spring, distance, magnification }}>{children}</DockContext.Provider>
      </motion.div>
    </motion.div>
  );
}

function DockItem({ children, className, label, onSelect, active = false, disabled = false }: DockItemProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { distance, magnification, mouseX, spring } = useDock();
  const isHovered = useMotionValue(0);

  const mouseDistance = useTransform(mouseX, (value) => {
    const box = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return value - box.x - box.width / 2;
  });
  const widthTransform = useTransform(mouseDistance, [-distance, 0, distance], [REST_SIZE, magnification, REST_SIZE]);
  const width = useSpring(widthTransform, spring);

  function onKeyDown(event: KeyboardEvent) {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    if (!disabled) onSelect?.();
  }

  return (
    <motion.div
      ref={ref}
      style={{ width }}
      onHoverStart={() => isHovered.set(1)}
      onHoverEnd={() => isHovered.set(0)}
      onFocus={() => isHovered.set(1)}
      onBlur={() => isHovered.set(0)}
      onClick={() => !disabled && onSelect?.()}
      onKeyDown={onKeyDown}
      className={cn('relative inline-flex items-center justify-center', className)}
      tabIndex={disabled ? -1 : 0}
      role="button"
      aria-label={label}
      aria-disabled={disabled || undefined}
      aria-current={active ? 'page' : undefined}
    >
      {Children.map(children, (child) => cloneElement(child as ReactElement<Record<string, unknown>>, { width, isHovered }))}
    </motion.div>
  );
}

function DockLabel({ children, className, ...rest }: DockPartProps) {
  const isHovered = (rest as Record<string, unknown>).isHovered as MotionValue<number>;
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => isHovered.on('change', (latest) => setIsVisible(latest === 1)), [isHovered]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 0 }}
          animate={{ opacity: 1, y: -10 }}
          exit={{ opacity: 0, y: 0 }}
          transition={{ duration: 0.2 }}
          className={cn('absolute -top-6 left-1/2 w-fit rounded-md px-2 py-0.5 text-xs whitespace-pre', className)}
          role="tooltip"
          style={{ x: '-50%' }}
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function DockIcon({ children, className, ...rest }: DockPartProps) {
  const width = (rest as Record<string, unknown>).width as MotionValue<number>;
  const halfWidth = useTransform(width, (value) => value / 2);
  return (
    <motion.div style={{ width: halfWidth }} className={cn('flex items-center justify-center', className)}>
      {children}
    </motion.div>
  );
}

export { Dock, DockIcon, DockItem, DockLabel };
