package com.gencior.triton.http;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManagerFactory;

import com.gencior.triton.config.TritonClientConfig;

/**
 * Factory for creating {@link SSLContext} instances from PEM certificate files.
 * Supports one-way TLS (server verification) and mutual TLS (mTLS).
 *
 * @author sachachoumiloff
 * @since 1.1.0
 */
final class SslContextFactory {

    private SslContextFactory() {}

    /**
     * Creates an SSLContext from the TLS configuration in TritonClientConfig.
     *
     * @param config the client configuration containing TLS settings
     * @return a configured SSLContext
     * @throws Exception if certificate loading or SSL initialization fails
     */
    static SSLContext createSslContext(TritonClientConfig config) throws Exception {
        TrustManagerFactory tmf = null;
        KeyManagerFactory kmf = null;

        if (config.getTrustCertFile() != null) {
            KeyStore trustStore = KeyStore.getInstance(KeyStore.getDefaultType());
            trustStore.load(null, null);
            List<X509Certificate> certs = loadCertificates(config.getTrustCertFile());
            for (int i = 0; i < certs.size(); i++) {
                trustStore.setCertificateEntry("ca-" + i, certs.get(i));
            }
            tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            tmf.init(trustStore);
        }

        if (config.getClientCertFile() != null && config.getClientKeyFile() != null) {
            List<X509Certificate> clientCerts = loadCertificates(config.getClientCertFile());
            PrivateKey privateKey = loadPrivateKey(config.getClientKeyFile());

            KeyStore keyStore = KeyStore.getInstance(KeyStore.getDefaultType());
            keyStore.load(null, null);
            keyStore.setKeyEntry("client", privateKey, new char[0],
                    clientCerts.toArray(new Certificate[0]));
            kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
            kmf.init(keyStore, new char[0]);
        }

        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(
                kmf != null ? kmf.getKeyManagers() : null,
                tmf != null ? tmf.getTrustManagers() : null,
                null
        );
        return sslContext;
    }

    private static List<X509Certificate> loadCertificates(File pemFile) throws Exception {
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        byte[] pemBytes = readPemContent(pemFile, "CERTIFICATE");
        // The file may contain multiple certificates
        String pemContent = new String(java.nio.file.Files.readAllBytes(pemFile.toPath()));
        List<X509Certificate> certs = new ArrayList<>();
        String[] blocks = pemContent.split("-----END CERTIFICATE-----");
        for (String block : blocks) {
            int begin = block.indexOf("-----BEGIN CERTIFICATE-----");
            if (begin >= 0) {
                String base64 = block.substring(begin + "-----BEGIN CERTIFICATE-----".length())
                        .replaceAll("\\s", "");
                byte[] decoded = Base64.getDecoder().decode(base64);
                X509Certificate cert = (X509Certificate) cf.generateCertificate(
                        new ByteArrayInputStream(decoded));
                certs.add(cert);
            }
        }
        return certs;
    }

    private static PrivateKey loadPrivateKey(File pemFile) throws Exception {
        String pemContent = new String(java.nio.file.Files.readAllBytes(pemFile.toPath()));
        // Support both PKCS#8 and traditional RSA/EC key formats
        String base64;
        String algorithm = "RSA";

        if (pemContent.contains("-----BEGIN PRIVATE KEY-----")) {
            base64 = extractBase64(pemContent, "PRIVATE KEY");
        } else if (pemContent.contains("-----BEGIN RSA PRIVATE KEY-----")) {
            base64 = extractBase64(pemContent, "RSA PRIVATE KEY");
        } else if (pemContent.contains("-----BEGIN EC PRIVATE KEY-----")) {
            base64 = extractBase64(pemContent, "EC PRIVATE KEY");
            algorithm = "EC";
        } else {
            throw new IllegalArgumentException("Unsupported private key format in " + pemFile);
        }

        byte[] decoded = Base64.getDecoder().decode(base64);
        KeyFactory kf = KeyFactory.getInstance(algorithm);
        return kf.generatePrivate(new java.security.spec.PKCS8EncodedKeySpec(decoded));
    }

    private static String extractBase64(String pem, String type) {
        String begin = "-----BEGIN " + type + "-----";
        String end = "-----END " + type + "-----";
        int startIdx = pem.indexOf(begin) + begin.length();
        int endIdx = pem.indexOf(end);
        return pem.substring(startIdx, endIdx).replaceAll("\\s", "");
    }

    private static byte[] readPemContent(File file, String type) throws IOException {
        // Utility kept for potential future use
        return java.nio.file.Files.readAllBytes(file.toPath());
    }
}
