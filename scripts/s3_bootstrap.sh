#!/usr/bin/env bash
set -euo pipefail
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_DEFAULT_REGION:-us-east-1}

for ENV in dev stg prod; do
  BUCKET="screenalytics-${ENV}-${ACCOUNT_ID}"
  echo "Ensuring bucket $BUCKET"
  if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    if [ "$REGION" = "us-east-1" ]; then
      aws s3api create-bucket --bucket "$BUCKET"
    else
      aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION"
    fi
  fi
  aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration '{
    "Rules":[
      {"ID":"videos-retention","Prefix":"raw/videos/","Status":"Enabled","Transitions":[
        {"Days":30,"StorageClass":"STANDARD_IA"},
        {"Days":90,"StorageClass":"GLACIER_IR"}
      ]},
      {"ID":"faces-expire","Prefix":"artifacts/faces/","Status":"Enabled","Expiration":{"Days":180}}
    ]}'
  echo "Bucket $BUCKET configured."
done
