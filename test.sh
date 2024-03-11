min=10
max=20
echo $(($RANDOM%($max-$min+1)+$min))
