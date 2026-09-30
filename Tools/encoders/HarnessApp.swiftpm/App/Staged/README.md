# Staged inputs for an iPhone run

The app copies this folder into its bundle. `Tools/encoders/stage_harness.sh` fills it with the
reference fixtures, the Verdict and Laya tokenizers and the converted Core ML packages, because an
iPhone cannot read the Mac's files. Git keeps only this README. The staged files are model weights
and tokenizer files, which never belong in the repository.
