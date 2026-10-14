"""命令列:python cli.py input.mp4 output.mp4 --stabilize --dehaze --enhance"""
import argparse
import sys

from pipeline import Options, process_video


def main() -> int:
    p = argparse.ArgumentParser(description="影片修復:提高畫質 / 除霧 / 防手震")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--enhance", action="store_true", help="提高畫質(不改變色彩的細節增強)")
    p.add_argument("--enhance-level", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--dehaze", action="store_true", help="暗通道先驗除霧")
    p.add_argument("--dehaze-strength", type=float, default=0.6)
    p.add_argument("--stabilize", action="store_true", help="防手震(含果凍效應修正)")
    p.add_argument("--stabilize-level", choices=["standard", "strong"], default="standard")
    p.add_argument("--zoom", type=float, default=0, help="防手震裁切放大倍率;0 = 自動")
    a = p.parse_args()

    opt = Options(enhance=a.enhance, enhance_level=a.enhance_level, dehaze=a.dehaze,
                  dehaze_strength=a.dehaze_strength, stabilize=a.stabilize,
                  stabilize_level=a.stabilize_level, zoom=a.zoom)
    if not (opt.enhance or opt.dehaze or opt.stabilize):
        print("請至少指定 --enhance / --dehaze / --stabilize 其中一項", file=sys.stderr)
        return 2

    def report(stage: str, frac: float) -> None:
        print(f"\r{stage} {frac:6.1%}", end="", flush=True)

    process_video(a.src, a.dst, opt, report)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
