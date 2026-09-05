# ============================================================
#   Разовый скрипт. Запустить один раз ПОСЛЕ применения
#   01_supabase_migration.sql:
#
#       python 02_seed_legacy_presets.py
#
#   Заливает старые 12 python-фильтров в таблицу filter_presets
#   как стартовые рецепты (is_legacy=true). Если запустить повторно —
#   не задублирует (проверяет по имени).
#
#   Нужны те же переменные окружения, что и у основного приложения:
#   SUPABASE_URL, SUPABASE_SERVICE_KEY (либо возьмите их из Infisical
#   вручную и подставьте здесь же временно).
# ============================================================
import os
from supabase import create_client
from filters_engine import LEGACY_PRESETS

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
    raise SystemExit(
        "Задайте SUPABASE_URL и SUPABASE_SERVICE_KEY в переменных окружения "
        "перед запуском (те же значения, что в Infisical/Render)."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

existing = supabase.table("filter_presets").select("name").execute()
existing_names = {r["name"] for r in existing.data}

inserted = 0
for preset in LEGACY_PRESETS:
    if preset["name"] in existing_names:
        print(f"пропущено (уже есть): {preset['name']}")
        continue
    supabase.table("filter_presets").insert(preset).execute()
    inserted += 1
    print(f"добавлено: {preset['name']}")

print(f"\nГотово. Добавлено новых рецептов: {inserted}")
