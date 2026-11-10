export const cn = (...names: Array<string | false | null | undefined>) => names.filter(Boolean).join(' ');
