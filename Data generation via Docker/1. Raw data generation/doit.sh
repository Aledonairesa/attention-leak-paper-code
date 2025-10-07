websites_file="$1"
calls_per_website="$2"

echo "Using websites file $1 and making $calls_per_website calls per website"

# Read each website from the websites file
while read website
do
	echo "Processing: $website"

	# For each website, run the specified number of calls
	for a in `seq 1 $calls_per_website`
	do
		# Count currently running Docker-related processed
		process_count=$(ps aux | grep -v grep | grep -c docker)

		# If too many running at once, wait
		if [ "$process_count" -gt 6 ]; then
			sleep 50
		else

			echo "Num:$a"

			# Generate unique container name
			num=`echo "$website-$RANDOM-$RANDOM-$RANDOM" |md5sum`

			# Run run.sh with unique name in the background
			echo "Running: bash run.sh $website-$num"
                	bash run.sh $website-$num & 
		fi
	done
done < "$websites_file"



