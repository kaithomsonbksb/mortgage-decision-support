from src.data import TARGET, load_data


def show_summary(df):
    print("Summary Statistics")
    print(df.describe().T.round(2).to_string())
    print()


def check_missing_values(df):
    print("Missing Values vs Target")

    null_cols = df.columns[df.isna().any()]
    if null_cols.empty:
        print("No missing values found")
        return

    for col in null_cols:
        missing = df[col].isna()
        count = missing.sum()
        mean_missing = df.loc[missing, TARGET].mean()
        mean_present = df.loc[~missing, TARGET].mean()

        print(
            f"{col}: {count} missing (Target mean -> "
            f"missing: {mean_missing:.3f} - present: {mean_present:.3f})"
        )


def main():
    df = load_data()
    show_summary(df)
    check_missing_values(df)


if __name__ == "__main__":
    main()
