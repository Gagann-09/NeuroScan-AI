"""
Configuration Security Tests

Tests verify that application configuration:
- Accepts required credential environment variables correctly
- Fails clearly when required production credentials are missing
- Does not silently fall back to insecure default credentials
- Preserves non-secret configuration behavior
"""
import os
import pytest
from pydantic import ValidationError


class TestConfigSecurity:
    """Test security-sensitive configuration behavior."""

    def test_database_url_required(self, monkeypatch):
        """DATABASE_URL must be provided - no default allowed."""
        # Remove any existing DATABASE_URL
        monkeypatch.delenv("DATABASE_URL", raising=False)
        
        # Should fail validation when Settings is instantiated
        from app.core.config import Settings
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        # Verify the error is about DATABASE_URL
        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "DATABASE_URL" for err in errors)

    def test_minio_access_key_required(self, monkeypatch):
        """MINIO_ACCESS_KEY must be provided - no default allowed."""
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        # Provide DATABASE_URL to isolate MINIO test
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
        
        from app.core.config import Settings
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "MINIO_ACCESS_KEY" for err in errors)

    def test_minio_secret_key_required(self, monkeypatch):
        """MINIO_SECRET_KEY must be provided - no default allowed."""
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "testkey")
        
        from app.core.config import Settings
        with pytest.raises(ValidationError) as exc_info:
            Settings()
        
        errors = exc_info.value.errors()
        assert any(err["loc"][0] == "MINIO_SECRET_KEY" for err in errors)

    def test_valid_credentials_accepted(self, monkeypatch):
        """Valid credentials from environment are accepted."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "testaccess")
        monkeypatch.setenv("MINIO_SECRET_KEY", "testsecret")
        
        from app.core.config import Settings
        settings = Settings()
        
        assert settings.DATABASE_URL == "postgresql://user:pass@localhost:5432/testdb"
        assert settings.MINIO_ACCESS_KEY == "testaccess"
        assert settings.MINIO_SECRET_KEY == "testsecret"

    def test_no_insecure_fallback_database_url(self, monkeypatch):
        """No silent fallback to known default password."""
        # Provide a URL with the old default password - should be accepted but not as default
        monkeypatch.setenv("DATABASE_URL", "postgresql://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "testaccess")
        monkeypatch.setenv("MINIO_SECRET_KEY", "testsecret")
        
        from app.core.config import Settings
        settings = Settings()
        
        # Should accept the explicitly provided URL
        assert settings.DATABASE_URL == "postgresql://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core"
        # But the default should NOT be present when not explicitly set
        # (This is tested in test_database_url_required)

    def test_non_secret_defaults_preserved(self, monkeypatch):
        """Non-sensitive configuration defaults are preserved."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "testaccess")
        monkeypatch.setenv("MINIO_SECRET_KEY", "testsecret")
        
        from app.core.config import Settings
        settings = Settings()
        
        # Non-secret defaults should still work
        assert settings.MINIO_ENDPOINT == "localhost:9000"
        assert settings.MINIO_SECURE is False
        assert settings.REDIS_URL == "redis://localhost:6379/0"
        assert settings.APP_TITLE == "NeuroScan AI API"
        assert settings.APP_VERSION == "1.0.0"
        assert settings.CORS_ORIGINS == ["*"]

    def test_get_settings_cached(self, monkeypatch):
        """get_settings returns cached instance."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/test")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "testaccess")
        monkeypatch.setenv("MINIO_SECRET_KEY", "testsecret")
        
        from app.core.config import get_settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same cached instance
        assert settings1 is settings2

    def test_get_settings_clear_error_on_missing_creds(self, monkeypatch):
        """get_settings provides clear error message for missing credentials."""
        # Clear the lru_cache by importing fresh module
        import importlib
        import app.core.config
        importlib.reload(app.core.config)
        
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        
        from app.core.config import get_settings
        
        with pytest.raises(RuntimeError) as exc_info:
            get_settings()
        
        error_msg = str(exc_info.value)
        assert "Missing required security credentials" in error_msg
        assert "DATABASE_URL" in error_msg
        assert "MINIO_ACCESS_KEY" in error_msg
        assert "MINIO_SECRET_KEY" in error_msg


class TestConfigDevelopmentWorkflow:
    """Test that local development workflow is not broken."""

    def test_docker_compose_environment_works(self, monkeypatch):
        """Simulate docker-compose environment variables."""
        # These are the values docker-compose.yml provides
        monkeypatch.setenv("DATABASE_URL", "postgresql://neuroscan_admin:secure_password_123@db:5432/neuroscan_core")
        monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "admin")
        monkeypatch.setenv("MINIO_SECRET_KEY", "secure_password_123")
        monkeypatch.setenv("MINIO_SECURE", "False")
        monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
        
        from app.core.config import Settings
        settings = Settings()
        
        assert settings.DATABASE_URL == "postgresql://neuroscan_admin:secure_password_123@db:5432/neuroscan_core"
        assert settings.MINIO_ENDPOINT == "minio:9000"
        assert settings.MINIO_ACCESS_KEY == "admin"
        assert settings.MINIO_SECRET_KEY == "secure_password_123"
        assert settings.MINIO_SECURE is False
        assert settings.REDIS_URL == "redis://redis:6379/0"

    def test_env_file_supported(self, monkeypatch, tmp_path):
        """Configuration can be loaded from .env file."""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text("""
DATABASE_URL=postgresql://test:test@localhost:5432/testdb
MINIO_ACCESS_KEY=fromenv
MINIO_SECRET_KEY=fromenvsecret
MINIO_ENDPOINT=custom:9000
""")
        
        # Remove environment variables so .env file takes precedence
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        
        # Change to temp directory and test
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Need to reimport to pick up new .env
            import importlib
            import app.core.config
            importlib.reload(app.core.config)
            
            from app.core.config import Settings
            settings = Settings()
            
            assert settings.DATABASE_URL == "postgresql://test:test@localhost:5432/testdb"
            assert settings.MINIO_ACCESS_KEY == "fromenv"
            assert settings.MINIO_SECRET_KEY == "fromenvsecret"
            assert settings.MINIO_ENDPOINT == "custom:9000"
        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])