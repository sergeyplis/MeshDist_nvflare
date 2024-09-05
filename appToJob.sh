#!/bin/bash

# Create the necessary directory structure
mkdir -p jobs/job/app/config

# Copy the app/config/ folder recursively to jobs/job/app/config
cp -r app/config/* jobs/job/app/config/
echo "The 'app/config/' folder has been successfully copied to 'jobs/job/app/config/'."

# Create and write the JSON content to meta.json in jobs/job/
cat << EOF > jobs/job/meta.json
{
  "name": "my_job",
  "resource_spec": {},
  "min_clients": 2,
  "deploy_map": {
    "app": [
      "@ALL"
    ]
  }
}
EOF

echo "meta.json has been successfully created in jobs/job/."