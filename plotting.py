# --- save data ---
    data = dict(
        betas=betas,
        betas_distortions=betas_distortions,
        supremum_distortions=supremum_distortions,
        config=vars(args),
    )
    data_path = os.path.join(output_dir, 'results.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved results to {data_path}")

    # --- plots ---
    # one-shot Borda; iterative Borda; one-shot Copeland; iterative Copeland; iterative ML with argmax; iterative ML with nonzero; iterative ML
    method_style = {
        'borda':            dict(color='C0', marker='o', linestyle='-',  label='one-shot Borda'),
        'borda_peeling':    dict(color='C1', marker='s', linestyle='--', label='iterative Borda'),
        'copeland':         dict(color='C2', marker='^', linestyle='-',  label='one-shot Copeland'),
        'copeland_peeling': dict(color='C3', marker='D', linestyle='--', label='iterative Copeland'),
        'ml_argmax':        dict(color='C4', marker='P', linestyle='-',  label='iterative ML with argmax'),
        'ml_nonzero':       dict(color='C5', marker='*', linestyle='--', label='iterative ML with nonzero'),
        'ml_sampling':      dict(color='C6', marker='h', linestyle='-',  label='iterative ML', linewidth=2),
    }

    title_suffix = f'(M={pw.M}, {args.num_samples} samples/round, {args.num_rounds} rounds)'
    small_betas = betas[betas <= 3.0]
    large_betas = betas[betas > 3.0]

    plots = [
        (betas,       betas_distortions,   'Fixed-weight distortion vs beta',             'fixed_weight_distortion.png'),
        (betas,       supremum_distortions, 'Supremum distortion vs beta',                 'supremum_distortion.png'),
        (small_betas, betas_distortions,   'Fixed-weight distortion vs beta (small beta)', 'fixed_weight_distortion_small_beta.png'),
        (small_betas, supremum_distortions, 'Supremum distortion vs beta (small beta)',    'supremum_distortion_small_beta.png'),
        (large_betas, betas_distortions,   'Fixed-weight distortion vs beta (large beta)', 'fixed_weight_distortion_large_beta.png'),
        (large_betas, supremum_distortions, 'Supremum distortion vs beta (large beta)',    'supremum_distortion_large_beta.png'),
    ]