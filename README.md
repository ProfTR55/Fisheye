# Fisheye Workspace

Bu klasor artik kaynak kod, veri ve ciktilari ayri tutacak sekilde duzenlendi.

## Ana Klasorler

- `fisheye_zernike_v2/`: aktif yeni proje. Zernike-first model, poly4 baseline,
  CLI, Hough bootstrap ve testler burada.
- `legacy_poly4/`: ilk/onceki poly4 tabanli pipeline. Eski komutlari karsilastirma
  icin burada sakliyoruz.
- `data/raw/`: ham input goruntuleri.
- `data/annotations/`: manuel cizgi JSON dosyalari ve notlar.
- `outputs/`: uretilmis debug klasorleri, rectified goruntuler ve karsilastirma
  ciktilari.
- `archive/`: eski deneme scriptleri, snapshotlar ve cache dosyalari.

## Aktif Projeyi Calistirma

```powershell
cd fisheye_zernike_v2

python -m fisheye_zernike.cli `
  --input ..\data\raw\20260421_150528_924.jpg.jpeg `
  --output out_zernike.jpg `
  --debug-dir debug_zernike `
  --auto-lines `
  --model zernike4 `
  --compare-models `
  --min-edge-angle 85 `
  --max-edge-angle 145 `
  --theta-bound-reg 120
```

## Manuel Secim

```powershell
cd fisheye_zernike_v2

python -m fisheye_zernike.cli `
  --input ..\data\raw\20260421_150528_924.jpg.jpeg `
  --output unused.jpg `
  --debug-dir debug_pick `
  --annotate-manual ..\data\annotations\manual_150528.json `
  --annotate-only
```

Sonra:

```powershell
python -m fisheye_zernike.cli `
  --input ..\data\raw\20260421_150528_924.jpg.jpeg `
  --output out_manual_zernike.jpg `
  --debug-dir debug_manual_zernike `
  --manual-lines ..\data\annotations\manual_150528.json `
  --model zernike4 `
  --compare-models
```
