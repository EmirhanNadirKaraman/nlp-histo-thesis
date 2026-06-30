# Environment Variable Loading - Simplified! ✨

## What Changed

**Before:** Every script had to manually load the `.env` file:
```python
# EVERY script needed this boilerplate
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass
```

**After:** It's automatic! Just import from database:
```python
# The .env file is loaded automatically
from database import get_db_connection
```

---

## How It Works

The `.env` file is now loaded at the **module level** in `db_connection.py`:

```python
# database/db_connection.py
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Now DB_CONFIG will use the loaded environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    ...
}
```

When **anyone** imports from the `database` package, the `.env` file is automatically loaded first.

---

## Benefits

### ✅ **Simpler Scripts**
No more repetitive boilerplate in every file!

**Before:**
```python
# Every script had 15+ lines of dotenv loading
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from database import get_db_connection  # Finally!
```

**After:**
```python
# Just import!
from database import get_db_connection  # .env loaded automatically
```

### ✅ **Can't Forget**
You can't accidentally forget to load the `.env` file - it happens automatically.

### ✅ **Centralized**
One place to maintain the dotenv loading logic instead of copying it everywhere.

### ✅ **Same Behavior**
Works exactly the same way, just cleaner:
- Looks for `.env` in project root
- Falls back to environment variables if .env not found
- Gracefully handles missing python-dotenv package

---

## What You Need to Do

**Nothing!** It just works now.

```python
# Any of these will automatically load .env:
from database import get_db_connection
from database import Document, TextElement
# (DB ingestion now lives in pipeline/stages/pdf_text_extraction/outputs/db_ingester.py)

# .env is already loaded by the time you import
db = get_db_connection()  # Uses credentials from .env
```

---

## For Script Authors

If you're writing new scripts that use the database:

**Don't do this anymore:**
```python
# ❌ OLD WAY - Don't do this
from dotenv import load_dotenv
load_dotenv()

from database import get_db_connection
```

**Just do this:**
```python
# ✅ NEW WAY - Simple!
from database import get_db_connection
```

---

## Testing

To verify it works:

```bash
# 1. Create a .env file (copy from .env.example at the repo root, which also
#    carries the summarization API keys OPENAI_API_KEY / GOOGLE_API_KEY /
#    ANTHROPIC_API_KEY — only the DB_* vars are needed for the database layer)
cat > .env << EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nlp_histo
DB_USER=postgres
DB_PASSWORD=test123
EOF

# 2. Run any database script (anything that imports the `database` package)
alembic current

# It will automatically use your .env settings!
```

---

## Technical Details

### Load Order
1. Python imports `database` package
2. `database/__init__.py` imports from `db_connection`
3. `db_connection.py` loads `.env` file (module-level code)
4. `DB_CONFIG` is created using loaded environment variables
5. Your code runs with correct configuration

### Environment Variable Precedence
1. **Actual environment variables** (highest priority)
2. **.env file** (loaded by dotenv)
3. **Default values** (hardcoded fallbacks)

Example:
```bash
# If you set an env var explicitly, it overrides .env
export DB_NAME=custom_db
alembic current  # Uses 'custom_db', not the .env value
```

---

## Summary

**One line of code** in `db_connection.py` eliminated **15+ lines** from every other script. That's good software engineering! 🎉

**Files simplified** (historical note — `setup_db.py` and `ingest.py` have
since been removed from `database/`; the schema is now Alembic-managed and
ingestion lives in the pipeline):
- ✅ all scripts that import from `database` - no longer need dotenv loading

**Centralized in:**
- ✅ `db_connection.py` - loads .env once, automatically
