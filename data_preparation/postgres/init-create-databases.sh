#!/bin/bash
set -e  # Exit on any error

# Optional isolation suffix (matches `prepare_data.py --out_suffix`). When given, this
# creates suffixed copies (e.g. clearscope_e3_test_for_opensource) instead of touching the
# originals, so you can validate the pipeline without overwriting existing databases.
#   ./init-create-databases.sh                       # original names
#   ./init-create-databases.sh _Test_for_OpenSource  # suffixed, isolated names
SUFFIX="${1:-}"

echo "Starting database and table creation (suffix='${SUFFIX}')..."

# Loop over datasets
for dataset in clearscope_e3 cadets_e3 theia_e3 clearscope_e5 cadets_e5 theia_e5
do
    # Append the suffix, then lowercase (unquoted CREATE DATABASE folds to lowercase; this
    # must match the lowercased name prepare_data.py connects with).
    DATASET_NAME=$(echo "${dataset}${SUFFIX}" | tr '[:upper:]' '[:lower:]')
    echo "Creating database and tables for: $DATASET_NAME"

    # PostgreSQL commands
    psql -U postgres <<EOF
CREATE DATABASE $DATASET_NAME;
\c $DATASET_NAME;

CREATE TABLE event_table (
    src_node VARCHAR,
    src_index_id VARCHAR,
    operation VARCHAR,
    dst_node VARCHAR,
    dst_index_id VARCHAR,
    event_uuid VARCHAR NOT NULL,
    timestamp_rec BIGINT,
    _id SERIAL PRIMARY KEY
);
ALTER TABLE event_table OWNER TO postgres;
CREATE UNIQUE INDEX event_table__id_uindex ON event_table (_id);
GRANT DELETE, INSERT, REFERENCES, SELECT, TRIGGER, TRUNCATE, UPDATE ON event_table TO postgres;

CREATE TABLE file_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    path VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE file_node_table OWNER TO postgres;

CREATE TABLE netflow_node_table (
    node_uuid VARCHAR NOT NULL,
    hash_id VARCHAR NOT NULL,
    src_addr VARCHAR,
    src_port VARCHAR,
    dst_addr VARCHAR,
    dst_port VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE netflow_node_table OWNER TO postgres;

CREATE TABLE subject_node_table (
    node_uuid VARCHAR,
    hash_id VARCHAR,
    path VARCHAR,
    cmd VARCHAR,
    index_id BIGINT,
    PRIMARY KEY (node_uuid, hash_id)
);
ALTER TABLE subject_node_table OWNER TO postgres;
EOF

    echo "Database '$DATASET_NAME' and tables created successfully!"
done

echo "All databases and tables created successfully!"