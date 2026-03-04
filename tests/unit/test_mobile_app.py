"""Tests for the mobile app CLI and model manager."""

from __future__ import annotations

from mobile_app.src.model_manager import ModelInfo, ModelManager


class TestModelManagerCheckModelsExist:
    """Tests for ModelManager.check_models_exist with mock directories."""

    def test_all_missing_when_empty_dir(self, tmp_path):
        manager = ModelManager(str(tmp_path))
        all_present, missing = manager.check_models_exist()
        assert all_present is False
        assert len(missing) == 4

    def test_all_present_when_files_exist(self, tmp_path):
        for _, rel_path in ModelManager.REQUIRED_MODELS:
            full_path = tmp_path / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(b"\x00" * 100)

        manager = ModelManager(str(tmp_path))
        all_present, missing = manager.check_models_exist()
        assert all_present is True
        assert missing == []

    def test_partial_models(self, tmp_path):
        # Create only the classifier model
        rel_path = ModelManager.REQUIRED_MODELS[0][1]
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(b"\x00" * 100)

        manager = ModelManager(str(tmp_path))
        all_present, missing = manager.check_models_exist()
        assert all_present is False
        assert len(missing) == 3
        assert "classifier" not in missing


class TestModelManagerGetModelInfo:
    """Tests for ModelManager.get_model_info."""

    def test_returns_all_required_models(self, tmp_path):
        manager = ModelManager(str(tmp_path))
        info = manager.get_model_info()
        assert len(info) == 4
        assert "classifier" in info
        assert "extractor_arztbesuch" in info
        assert "extractor_reisekosten" in info
        assert "extractor_lieferschein" in info

    def test_model_info_structure(self, tmp_path):
        manager = ModelManager(str(tmp_path))
        info = manager.get_model_info()
        for mi in info.values():
            assert isinstance(mi, ModelInfo)
            assert isinstance(mi.name, str)
            assert isinstance(mi.path, str)
            assert isinstance(mi.size_mb, float)
            assert isinstance(mi.exists, bool)

    def test_existing_model_has_nonzero_size(self, tmp_path):
        rel_path = ModelManager.REQUIRED_MODELS[0][1]
        full_path = tmp_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(b"\x00" * 1024 * 1024)  # 1 MB

        manager = ModelManager(str(tmp_path))
        info = manager.get_model_info()
        assert info["classifier"].exists is True
        assert info["classifier"].size_mb == 1.0

    def test_missing_model_has_zero_size(self, tmp_path):
        manager = ModelManager(str(tmp_path))
        info = manager.get_model_info()
        assert info["classifier"].exists is False
        assert info["classifier"].size_mb == 0.0


class TestModelManagerTotalSize:
    """Tests for ModelManager.get_total_size_mb."""

    def test_total_size_zero_when_no_models(self, tmp_path):
        manager = ModelManager(str(tmp_path))
        assert manager.get_total_size_mb() == 0.0

    def test_total_size_sums_existing(self, tmp_path):
        for _, rel_path in ModelManager.REQUIRED_MODELS[:2]:
            full_path = tmp_path / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(b"\x00" * 1024 * 1024)  # 1 MB each

        manager = ModelManager(str(tmp_path))
        assert manager.get_total_size_mb() == 2.0


class TestAppCLIParsing:
    """Tests for CLI argument parsing."""

    def test_process_command(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["process", "test.png"])
        assert args.command == "process"
        assert args.image_path == "test.png"

    def test_info_command(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["info"])
        assert args.command == "info"
        assert args.models_dir == "edge_model"

    def test_info_command_custom_dir(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["info", "--models-dir", "/custom/path"])
        assert args.models_dir == "/custom/path"

    def test_batch_command(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["batch", "images/"])
        assert args.command == "batch"
        assert args.directory == "images/"
        assert args.output is None

    def test_batch_command_with_output(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["batch", "images/", "-o", "results.json"])
        assert args.output == "results.json"

    def test_demo_command(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.command == "demo"

    def test_no_command_returns_none(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_config_default(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["info"])
        assert args.config == "config.yaml"

    def test_config_custom(self):
        from mobile_app.src.app import build_parser

        parser = build_parser()
        args = parser.parse_args(["--config", "custom.yaml", "info"])
        assert args.config == "custom.yaml"
