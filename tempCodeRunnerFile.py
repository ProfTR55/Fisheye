    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img).save("mask.png")

    # mask sınırını panoramada kırmızı çiz
    out_dbg = out.copy()
    edge = mask ^ np.roll(mask, 1, axis=0) ^ np.roll(mask, 1, axis=1)
    out_dbg[edge] = np.array([255, 0, 0], dtype=np.float32)
    Image.fromarray(np.clip(out_dbg,0,255).astype(np.uint8)).save("panorama_edge.png")

