#!/bin/bash
name=$1
website=`echo $name|cut -d"-" -f1`
data_dir="data/$name"

# Ensure base data directory exists
mkdir -p data/

# Step 1: Run the Docker container if data hasn't already been collected
if [ ! -d "$data_dir" ]; then

	# Run docker container in detached mode
	docker run -d --name $name  ubuntu /root/generator/script.sh $website

	# Step 2: Wait so the machine doesn't saturate
	sleep 5 

	# Step 3: Copy the file from the container to the host
	mkdir -p "$data_dir"
	docker cp $name:/root/generator/data.pcap "$data_dir/"
	docker cp $name:/root/generator/log.txt "$data_dir/"

	tshark -r "$data_dir/data.pcap" \
	  -T fields -e frame.number -e frame.time -e ip.src -e ip.dst -e ip.proto \
	  -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport \
	  -e tcp.flags.str -e frame.len \
	  -E header=y -E separator=, -E quote=d -E occurrence=f \
	  > "$data_dir/data.csv"

	# Step 4: Stop and remove the container
	docker stop $name 
	docker rm $name

else
	echo "Data already exists for $name. Skipping."
fi
