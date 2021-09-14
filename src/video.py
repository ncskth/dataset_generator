

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Postprocess NRP DVS datasets")
    parser.add_argument("files", nargs="+", help="Datasets to parse")
    main(parser.parse_args())