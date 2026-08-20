create extension if not exists pgcrypto;

create table if not exists app_users (
    id uuid primary key default gen_random_uuid(),
    email text not null unique,
    username text not null unique,
    full_name text not null default '',
    password_hash text not null,
    role text not null default 'user',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists instrument_tags (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references app_users(id) on delete cascade,
    symbol text not null,
    role text not null default 'Watching',
    active boolean not null default true,
    notes text not null default '',
    shares numeric not null default 0,
    average_cost numeric not null default 0,
    book_cost numeric not null default 0,
    category text not null default '',
    purchase_date date,
    intent text not null default '',
    strategy text not null default '',
    watch_reason text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique(user_id, symbol)
);

create table if not exists price_alerts (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references app_users(id) on delete cascade,
    symbol text not null,
    role text not null default 'Watching',
    metric text not null default 'Price',
    operator text not null default 'Crossing',
    threshold numeric not null default 0,
    trigger text not null default 'Once only',
    expiration date,
    message text not null default '',
    notifications jsonb not null default '["In-app", "Toast"]'::jsonb,
    enabled boolean not null default true,
    created_at timestamptz not null default now(),
    last_value numeric,
    last_triggered_at timestamptz
);

create index if not exists idx_instrument_tags_user_active on instrument_tags(user_id, active);
create index if not exists idx_price_alerts_user_enabled on price_alerts(user_id, enabled);
