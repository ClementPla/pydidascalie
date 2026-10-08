"""Database schema for Didascalie files."""

SCHEMA = """
-- Project metadata (single row)
CREATE TABLE IF NOT EXISTS project (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    config JSON NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Labels
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    color TEXT NOT NULL,
    is_instance BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0
);

-- Sequences (every image belongs to a sequence, even if alone)
CREATE TABLE IF NOT EXISTS sequences (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    sort_order INTEGER DEFAULT 0
);

-- Frames (images within sequences)
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY,
    sequence_id INTEGER NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    relative_path TEXT,
    content_hash TEXT,
    embedded_data BLOB,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    reviewed BOOLEAN DEFAULT FALSE,
    UNIQUE(sequence_id, frame_index)
);

-- Annotations (per frame, per label)
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    encoding TEXT NOT NULL DEFAULT 'rle',
    mask_data BLOB NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, label_id)
);

-- Classifications
CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    task_name TEXT NOT NULL,
    selected_classes JSON NOT NULL,
    is_multilabel BOOLEAN DEFAULT FALSE,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, task_name)
);

-- Text descriptions
CREATE TABLE IF NOT EXISTS text_descriptions (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    label_name TEXT NOT NULL,
    content TEXT NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, label_name)
);

-- Registrations: a (reference, moving) frame pair of a sequence, with the
-- keypoint correspondences placed between them. Same tables as the application.
CREATE TABLE IF NOT EXISTS registrations (
    id INTEGER PRIMARY KEY,
    sequence_id INTEGER NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
    reference_frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    moving_frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    -- 9 floats for a 3x3 homography (row-major), JSON-encoded, or null.
    homography JSON,
    transform_type TEXT NOT NULL DEFAULT 'homography',
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_frame_id, moving_frame_id),
    CHECK (reference_frame_id != moving_frame_id)
);

-- Individual keypoint correspondences within a registration.
CREATE TABLE IF NOT EXISTS keypoint_pairs (
    id INTEGER PRIMARY KEY,
    registration_id INTEGER NOT NULL REFERENCES registrations(id) ON DELETE CASCADE,
    client_uuid TEXT NOT NULL,
    -- Image-native pixel coordinates, sub-pixel.
    ref_x REAL NOT NULL,
    ref_y REAL NOT NULL,
    moving_x REAL NOT NULL,
    moving_y REAL NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(registration_id, client_uuid)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_frames_sequence ON frames(sequence_id);
CREATE INDEX IF NOT EXISTS idx_annotations_frame ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_classifications_frame ON classifications(frame_id);
CREATE INDEX IF NOT EXISTS idx_text_descriptions_frame ON text_descriptions(frame_id);
CREATE INDEX IF NOT EXISTS idx_registrations_sequence ON registrations(sequence_id);
CREATE INDEX IF NOT EXISTS idx_registrations_ref_frame ON registrations(reference_frame_id);
CREATE INDEX IF NOT EXISTS idx_keypoint_pairs_registration ON keypoint_pairs(registration_id);
"""

# Newest schema version of the application this package reads and writes.
#
#   1  baseline
#   2  vector annotations
#   3  masks stored as `rle8` (a value per pixel)
#   4  user accounts: annotations, classifications, texts and reviews belong to
#      a user, and "reviewed" moves from `frames.reviewed` to `frame_reviews`
SCHEMA_VERSION = 4

# Version stamped into the files this package creates. They are written in the
# version 3 layout above, which every release of the application since `rle8`
# opens, and which the releases with user accounts upgrade on first open.
CREATE_SCHEMA_VERSION = 3

# The application's schema at version 4, copied from its source
# (`src-tauri/src/storage/schema.rs`). `enable_user_accounts` applies it to
# bring a project to that layout, the way the application does on first open.
SCHEMA_V4 = """
-- Project metadata (single row)
CREATE TABLE IF NOT EXISTS project (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    config JSON NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Labels
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    color TEXT NOT NULL,
    is_instance BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0
);

-- Sequences (every image belongs to a sequence, even if alone)
CREATE TABLE IF NOT EXISTS sequences (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    sort_order INTEGER DEFAULT 0
);

-- Frames (images within sequences)
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY,
    sequence_id INTEGER NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    relative_path TEXT,
    content_hash TEXT,
    embedded_data BLOB,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    reviewed BOOLEAN DEFAULT FALSE,
    UNIQUE(sequence_id, frame_index)
);

-- The people annotating this project. There is always at least one: the
-- account created with the project, or added to an older project on its first
-- open, which is an administrator and owns whatever was annotated before
-- accounts existed.
--
-- `password` is stored as typed. It keeps colleagues from picking each other's
-- account by mistake; it is not protection against someone who has the file.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL DEFAULT 'editor' CHECK (role IN ('admin', 'editor')),
    password TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Runs on every open, like the rest of this baseline, and does nothing once
-- the project has an account. It is what makes "there is always one" true, and
-- what the `DEFAULT 1` on the `user_id` columns below points at.
INSERT INTO users (id, name, role)
    SELECT 1, 'Admin', 'admin' WHERE NOT EXISTS (SELECT 1 FROM users);

-- Everything below this line and above `registrations` is annotated *by
-- someone*: one independent set per user, so two graders can label the same
-- frame without seeing or overwriting each other.
--
-- `user_id` defaults to 1, the account every project has. That default is what
-- a writer that predates accounts (the Python library, an old script) lands
-- on, and it keeps `INSERT`s without the column valid.
--
-- While a project is open in the app, each of these tables is shadowed by a
-- connection-local view of the same name that only shows the logged-in user's
-- rows — see `queries::install_user_scope`.

-- Annotations (per frame, per label, per user)
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
    encoding TEXT NOT NULL DEFAULT 'rle',
    mask_data BLOB NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, label_id, user_id)
);

-- Vector annotations (per frame, per label, per user): bezier paths / polygons
-- / polylines. `shapes` is a JSON array of VectorShape, owned and validated by
-- the frontend.
CREATE TABLE IF NOT EXISTS vector_annotations (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
    shapes JSON NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, label_id, user_id)
);
CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
    task_name TEXT NOT NULL,
    selected_classes JSON NOT NULL,
    is_multilabel BOOLEAN DEFAULT FALSE,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, task_name, user_id)
);
CREATE TABLE IF NOT EXISTS text_descriptions (
    id INTEGER PRIMARY KEY,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
    label_name TEXT NOT NULL,
    content TEXT NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, label_name, user_id)
);

-- Which frames each user has marked as reviewed. A row means reviewed; no row
-- means not. This replaces `frames.reviewed`, which could only say that
-- *someone* had: that column is still there so older files and tools keep
-- working, but it is read once, when a project is migrated, and never again.
CREATE TABLE IF NOT EXISTS frame_reviews (
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL DEFAULT 1 REFERENCES users(id) ON DELETE CASCADE,
    reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (frame_id, user_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS registrations (
    id INTEGER PRIMARY KEY,
    sequence_id INTEGER NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
    reference_frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    moving_frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    -- 9 floats for a 3x3 homography (row-major), JSON-encoded.
    -- NULL until fewer than 4 pairs have been placed.
    homography JSON,
    transform_type TEXT NOT NULL DEFAULT 'homography',
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reference_frame_id, moving_frame_id),
    CHECK (reference_frame_id != moving_frame_id)
);

-- Individual keypoint correspondences within a registration.
CREATE TABLE IF NOT EXISTS keypoint_pairs (
    id INTEGER PRIMARY KEY,
    registration_id INTEGER NOT NULL REFERENCES registrations(id) ON DELETE CASCADE,
    -- Stable string id from the frontend (crypto.randomUUID()).
    -- Lets the frontend round-trip its own ids without remapping.
    client_uuid TEXT NOT NULL,
    -- Image-native pixel coordinates. REAL not INTEGER — sub-pixel placement
    -- happens when the user drags points around at high zoom.
    ref_x REAL NOT NULL,
    ref_y REAL NOT NULL,
    moving_x REAL NOT NULL,
    moving_y REAL NOT NULL,
    -- For stable ordering in the UI list (matches insertion order).
    sort_order INTEGER NOT NULL DEFAULT 0,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(registration_id, client_uuid)
);


CREATE INDEX IF NOT EXISTS idx_frames_sequence ON frames(sequence_id);
CREATE INDEX IF NOT EXISTS idx_annotations_frame ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_vector_annotations_frame ON vector_annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_classifications_frame ON classifications(frame_id);
CREATE INDEX IF NOT EXISTS idx_text_descriptions_frame ON text_descriptions(frame_id);
CREATE INDEX IF NOT EXISTS idx_registrations_sequence
    ON registrations(sequence_id);
CREATE INDEX IF NOT EXISTS idx_registrations_ref_frame
    ON registrations(reference_frame_id);
CREATE INDEX IF NOT EXISTS idx_keypoint_pairs_registration
    ON keypoint_pairs(registration_id);
CREATE INDEX IF NOT EXISTS idx_frame_reviews_user ON frame_reviews(user_id);

-- A fitted segmentation head, so training survives closing the app.
--
-- One row: `id` is pinned to 1 and writes upsert over it. Keeping a history
-- would mean a UI to choose between models, and the useful question after
-- retraining is "use the new one", not "which of the seven".
--
-- Weights live here rather than in app data because class `i + 1` means
-- `label_order[i]` — ids that only exist in this project — so the model and the
-- labels that give it meaning stay together when the project is copied.
CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    meta JSON NOT NULL,
    weights BLOB NOT NULL,
    modified_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# The tables that became per-user in version 4, with the columns they had before.
V3_PER_USER_TABLES = {
    "annotations": "id, frame_id, label_id, encoding, mask_data, modified_at",
    "vector_annotations": "id, frame_id, label_id, shapes, modified_at",
    "classifications": "id, frame_id, task_name, selected_classes, is_multilabel, modified_at",
    "text_descriptions": "id, frame_id, label_name, content, modified_at",
}
