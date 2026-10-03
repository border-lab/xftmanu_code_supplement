FINAL-SUBMISSION NUMERICAL SUPPLEMENT — STAGED FOR THE EXISTING REPOSITORY

Target: https://github.com/border-lab/xftmanu_code_supplement
Published in the repository on 2026-10-02 as release v4.

decomposition_source_data.zip: complete generation 0–5 fitted symmetric weights,
ratios, apparent quantities, and plotting data for current Figure 4b–d,
Extended Data Figures 4/5, Supplementary Figure S17 and Tables S7/S8.
21,000 replicate-generation records; includes general-VT extraction code.

figure5_source_data.zip: original Figure 5 statistics, 200 replicates across
generations 0–5, the full identified 2-by-3 weight matrices, and reproduction
code. Wealth has no separate direct genetic component. Undefined ratios remain
explicitly undefined. Existing displayed numbers were preserved.

legacy_vt_source_data.zip: 4,608 original replicate-generation covariance
records for the VT conditions in ED1 and residual-correlation 0/0.1/0.2 in ED10.
Coverage is limited to those conditions. In GxE settings these are projections
on the direct genetic components, not assertions of full-genotype BLPs.

The Source_Data workbooks provide indexed numerical sheets. Supplementary_Tables
contains the existing S1–S4 phenotype definitions and empirical summary results.
No individual-level UK Biobank or Taiwan NHIRD participant data are included.

Extract each numerical archive into its own directory and follow its README
and reproduction script. Each archive contains its provenance and verification.
figure_layout_snapshot.zip preserves the local layout workspace and its saved
simulation inputs; its historical figure numbers precede the final split:
old Figure 1 -> current Figures 1/2; old Figure 2 -> current Figure 3;
old Figure 3 -> current Figure 4; old Figure 4 -> current Figure 5.
ED4 and ED5 were exchanged in the final revision. Final artwork is supplied
separately. Figure-style scripts retain their documented workspace assumptions.
artwork_sources/figure5 contains the original schematic SVG and the editable-text
orange/blue SVG/PDF used in the final Figure 5. Original vector math is preserved
as paths; the DNA icons retain their original raster textures.

The final font cleanup brings editable text in the main and ED vector PDFs to
5–7 pt and retains 7 pt panel letters. figure_style_sources includes the cleanup
script, its verification record, and pre_font_cleanup_artwork.zip containing its
original PDF inputs. This adapter retains the recorded workspace and Linux font
assumptions. Plot paths and values are unchanged. Figure5d's missing beta glyph
was restored from the executed mFigEdu notebook, cell36.

Abandoned partial recovery checkpoints and experimental document comparisons
are intentionally excluded. No new simulations were run for these exports.
