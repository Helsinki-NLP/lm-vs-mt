train_=0
val_=0
test_=0
for corpus in unpc opensubtitles; do
  corpus_total=0;
  for split in  val test train; do
    file=processed/docs.${corpus}.${split}.bpe
    lines=$(sed -e '/..\t$/d' $file | wc -l)
    echo $file $lines
    corpus_total=$(expr $corpus_total + $lines)
    if [ $split == "train" ]; then train_=$(expr $train_ + $lines); fi
    if [ $split == "val" ]; then val_=$(expr $val_ + $lines); fi
    if [ $split == "test" ]; then test_=$(expr $test_ + $lines); fi
  done
  echo $corpus $corpus_total
done
echo train $train_
echo val $val_
echo test $test_

echo total $(expr $train_ + $val_ + $test_)
