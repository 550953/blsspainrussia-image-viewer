-- Таблица разметки цен. Выполнить один раз в Supabase -> SQL Editor.

create table if not exists price_checks (
    id bigint generated always as identity primary key,
    filename text not null unique,      -- 20260830_181757_023_empty_empty.jpg
    url text not null,                  -- полный URL фото
    label text not null,                -- текущее (редактируемое) значение цены
    original_label text not null,       -- значение из исходного CSV
    comment text,                       -- заметка пользователя
    is_broken boolean not null default false,  -- флаг "чек забагован"
    updated_at timestamptz not null default now()
);

create index if not exists price_checks_filename_idx on price_checks (filename);
create index if not exists price_checks_is_broken_idx on price_checks (is_broken);

-- Автообновление updated_at при любом изменении строки
create or replace function set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists price_checks_set_updated_at on price_checks;
create trigger price_checks_set_updated_at
before update on price_checks
for each row execute function set_updated_at();

-- RLS включаем, но НЕ добавляем политик для anon/authenticated:
-- бэкенд обращается к таблице через service_role key (обходит RLS),
-- поэтому напрямую из браузера через anon key к этой таблице никто не достучится.
alter table price_checks enable row level security;
