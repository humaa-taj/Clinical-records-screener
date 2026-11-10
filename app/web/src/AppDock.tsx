import type { LucideIcon } from 'lucide-react';
import { Dock, DockIcon, DockItem, DockLabel } from '@/components/ui/dock';
import { cn } from '@/lib/utils';

export type DockEntry = {
  id: string;
  label: string;
  icon: LucideIcon;
  onSelect: () => void;
  active?: boolean;
  disabled?: boolean;
};

export default function AppDock({ entries }: { entries: DockEntry[] }) {
  return (
    <nav className="pointer-events-none fixed inset-x-0 bottom-3 z-20 flex justify-center" aria-label="Main">
      <Dock className="dock-panel items-end pb-2.5">
        {entries.map(({ id, label, icon: Icon, onSelect, active, disabled }) => (
          <DockItem key={id} label={label} onSelect={onSelect} active={active} disabled={disabled} className={cn('dock-item aspect-square rounded-full', active && 'dock-item-active', disabled && 'dock-item-disabled')}>
            <DockLabel className="dock-label">{label}</DockLabel>
            <DockIcon>
              <Icon className="h-full w-full" strokeWidth={1.7} aria-hidden="true" />
            </DockIcon>
          </DockItem>
        ))}
      </Dock>
    </nav>
  );
}
