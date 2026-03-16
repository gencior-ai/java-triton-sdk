#!/bin/bash
# Generates self-signed test certificates for TLS integration tests.
# Run from the dev/certs/ directory.
set -e
cd "$(dirname "$0")"

# CA (root authority)
openssl req -x509 -newkey rsa:2048 -keyout ca.key -out ca.pem -days 3650 -nodes -subj "/CN=TritonTestCA"

# Server certificate signed by CA (with SAN for localhost)
openssl req -newkey rsa:2048 -keyout server.key -out server.csr -nodes -subj "/CN=localhost"
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out server.pem -days 3650 \
  -extfile <(echo "subjectAltName=DNS:localhost,IP:127.0.0.1")

# Client certificate signed by CA
openssl req -newkey rsa:2048 -keyout client.key -out client.csr -nodes -subj "/CN=TritonTestClient"
openssl x509 -req -in client.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out client.pem -days 3650

# Wrong CA (not the one that signed the server cert)
openssl req -x509 -newkey rsa:2048 -keyout wrong-ca.key -out wrong-ca.pem -days 3650 -nodes -subj "/CN=WrongCA"

# Cleanup
rm -f *.csr *.srl

echo "Certificates generated successfully."
