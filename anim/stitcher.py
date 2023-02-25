import subprocess
from pathlib import Path

__all__ = ['stitch']

# ffmpeg -framerate 60 -pattern_type glob -i "frames/*.png" -c:v libx264 -pix_fmt yuv420p out.mp4

def stitch(framerate: int, framedir: Path, outpath: Path = Path('out.mp4'), overwrite: bool = False):
	args = []
	args += ['ffmpeg']
	args += [
		'-framerate', str(framerate),
		'-pattern_type', 'glob',
		'-i', str(framedir / '*.png'),
		'-c:v', 'libx264',
		'-pix_fmt', 'yuv420p',
	]
	if overwrite:
		args += ['-y']
	args += [str(outpath)]

	subprocess.run(args)