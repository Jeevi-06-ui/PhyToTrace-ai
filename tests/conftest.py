"""Keep the test suite offline unless a dedicated live test environment is supplied."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_local_storage(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        workspace = tmp_path_factory.mktemp("phytotrace")
        patch.setenv("PHYTOTRACE_SQLITE_PATH", str(workspace / "inspections.db"))
        patch.setenv("PHYTOTRACE_ALLOW_LOCAL_SQLITE", "1")
        if os.getenv("PHYTOTRACE_RUN_LIVE_TESTS") != "1":
            for name in ("PHYTOTRACE_ROTATED_CLOUD_CREDENTIALS", "MONGODB_URI",
                         "CLOUDINARY_API_SECRET", "GEMINI_API_KEY", "GEMINI_ASSISTANT_API_KEY",
                         "GEMINI_EXPLANATION_API_KEY", "R2_ACCOUNT_ID"):
                patch.delenv(name, raising=False)
        from phytotrace.api import main
        patch.setattr(main, "STORAGE_RAW", workspace / "raw")
        patch.setattr(main, "STORAGE_MASKS", workspace / "masks")
        patch.setattr(main, "STORAGE_OVERLAYS", workspace / "overlays")
        yield
