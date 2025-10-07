import pandas as pd
import ipaddress
import numpy as np

class FeatureExtractor:
    def __init__(self, data):
        # Extract the full dataframe
        self.full = pd.DataFrame.from_dict(data)
        # Extract the previous frames
        self.previous = self.full.iloc[:-1]
        # Extract the new frame
        self.new = self.full.iloc[-1:]
        # Define new row
        self.new_row = {}
    
    # frame.time features
    # ==============================================================================================
    def extract_time_to_last_frame(self):
        """
        The difference in time between the new frame's time and the time of
        the last frame of the previous set of frames.
        """
        self.new_row['time_to_last_frame'] = self.new['frame.time'].iloc[0] - self.previous['frame.time'].iloc[-1]
        
    # send features
    # ==============================================================================================
    def extract_send_in_last_send(self):
        """
        Whether or not the send column of the new frame and the last frame of
        the previous frames is the same or not.
        """
        self.new_row['send_in_last_send'] = int(self.new['send'].iloc[0] == self.previous['send'].iloc[-1])
        
    def extract_send_fraction_in_send_50(self):
        """
        Depending on whether the new frame is sending or receiving, calculates
        the fraction of sends/receives in the max 50 previous frames.
        """
        current_send = self.new['send'].iloc[0]
        last_50_previous_send = self.previous['send'].iloc[-50:]
        
        last_50_fraction_send = sum(last_50_previous_send)/len(last_50_previous_send)
        if current_send == 0:
            last_50_fraction_send = 1 - last_50_fraction_send
        
        self.new_row['send_fraction_in_send_50'] = last_50_fraction_send
        
    def extract_send_fraction_in_send_ALL(self):
        """
        Depending on whether the new frame is sending or receiving, calculates
        the fraction of sends/receives in the max 50 previous frames.
        """
        current_send = self.new['send'].iloc[0]
        last_50_previous_send = self.previous['send']
        
        last_50_fraction_send = sum(last_50_previous_send)/len(last_50_previous_send)
        if current_send == 0:
            last_50_fraction_send = 1 - last_50_fraction_send
        
        self.new_row['send_fraction_in_send_ALL'] = last_50_fraction_send
        
    def extract_current_send(self):
        """
        Whether the new frame's host is sending or receiving information.
        """
        self.new_row['current_send'] = self.new['send'].iloc[0]
        
    # asn features
    # ==============================================================================================
    def extract_country_in_country(self):
        """
        Whether or not the country of the new frame is in the country column of any 
        of the previous frames.
        """
        new_country = self.new['asn_country'].iloc[0]
        self.new_row['country_in_country'] = int(self.previous['asn_country'].str.contains(new_country).any())
    
    def extract_country_in_last_country(self):
        """
        Whether or not the country matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['country_in_last_country'] = int(self.new['asn_country'].iloc[0] == self.previous['asn_country'].iloc[-1])
    
    def extract_country_fraction_in_country_50(self):
        """
        Calculates the fraction of the same country in the max 50 previous frames.
        """
        new = self.new['asn_country'].iloc[0]
        
        previous = self.previous['asn_country'].iloc[-50:]
        total = len(previous)
        count = (previous == new).sum()

        if total > 0:
            self.new_row['country_fraction_in_country_50'] = count / total
        else:
            self.new_row['country_fraction_in_country_50'] = 0.0
    
    def extract_country_fraction_in_country_ALL(self):
        """
        Calculates the fraction of the same country in the max 50 previous frames.
        """
        new = self.new['asn_country'].iloc[0]
        
        previous = self.previous['asn_country']
        total = len(previous)
        count = (previous == new).sum()

        if total > 0:
            self.new_row['country_fraction_in_country_ALL'] = count / total
        else:
            self.new_row['country_fraction_in_country_ALL'] = 0.0
    
    def extract_desc_in_desc(self):
        """
        Whether or not the asn description of the new frame is in the asn description column of any 
        of the previous frames.
        """
        new_desc = self.new['asn_description'].iloc[0]
        self.new_row['desc_in_desc'] = int(self.previous['asn_description'].str.contains(new_desc).any())
    
    def extract_desc_in_last_desc(self):
        """
        Whether or not the asn description matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['desc_in_last_desc'] = int(self.new['asn_description'].iloc[0] == self.previous['asn_description'].iloc[-1])
    
    def extract_desc_fraction_in_desc_50(self):
        """
        Calculates the fraction of the same asn description in the max 50 previous frames.
        """
        new = self.new['asn_description'].iloc[0]
        
        previous = self.previous['asn_description'].iloc[-50:]
        total = len(previous)
        count = (previous == new).sum()

        if total > 0:
            self.new_row['desc_fraction_in_desc_50'] = count / total
        else:
            self.new_row['desc_fraction_in_desc_50'] = 0.0
            
    def extract_desc_fraction_in_desc_ALL(self):
        """
        Calculates the fraction of the same asn description in the max 50 previous frames.
        """
        new = self.new['asn_description'].iloc[0]
        
        previous = self.previous['asn_description']
        total = len(previous)
        count = (previous == new).sum()

        if total > 0:
            self.new_row['desc_fraction_in_desc_ALL'] = count / total
        else:
            self.new_row['desc_fraction_in_desc_ALL'] = 0.0
    
    def extract_network_in_network(self):
        """
        Whether or not the network name of the new frame is in the netowork column of any 
        of the previous frames.
        """
        new = self.new['network_name'].iloc[0]
        self.new_row['network_in_network'] = int(self.previous['network_name'].str.contains(new).any())
    
    def extract_network_in_last_network(self):
        """
        Whether or not the network name matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['network_in_last_network'] = int(self.new['network_name'].iloc[0] == self.previous['network_name'].iloc[-1])
    
    def extract_network_fraction_in_network_50(self):
        """
        Calculates the fraction of the same network name in the max 50 previous frames.
        """
        new = self.new['network_name'].iloc[0]
        
        previous = self.previous['network_name'].iloc[-50:]
        total = len(previous)
        count = (previous == new).sum()

        if total > 0:
            self.new_row['network_fraction_in_network_50'] = count / total
        else:
            self.new_row['network_fraction_in_network_50'] = 0.0
    
    def extract_network_fraction_in_network_ALL(self):
        """
        Calculates the fraction of the same network name in the max 50 previous frames.
        """
        new = self.new['network_name'].iloc[0]
        
        previous = self.previous['network_name']
        total = len(previous)
        count = (previous == new).sum()

        if total > 0:
            self.new_row['network_fraction_in_network_ALL'] = count / total
        else:
            self.new_row['network_fraction_in_network_ALL'] = 0.0
        
    # ip features
    # ==============================================================================================
    def extract_ip_in_ip(self):
        """
        Whether or not the IP of the new frame is in the ip column of any of
        any of the previous frames.
        """
        new_ip = self.new['ip'].iloc[0]
        self.new_row['ip_in_ip'] = int(self.previous['ip'].str.contains(new_ip).any())
    
    def extract_ip_in_last_ip(self):
        """
        Whether or not the IP address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['ip_in_last_ip'] = int(self.new['ip'].iloc[0] == self.previous['ip'].iloc[-1])
        
    # ip.src features
    # ==============================================================================================
    def extract_ipsrc_in_ipsrc(self):
        """
        Whether or not the source IP of the new frame is in the ip.src column
        of any of the previous frames.
        """
        new_ipsrc = self.new['ip.src'].iloc[0]
        self.new_row['ipsrc_in_ipsrc'] = int(self.previous['ip.src'].str.contains(new_ipsrc).any())
    
    def extract_ipsrc_in_ipdst(self):
        """
        Whether or not the source IP of the new frame is in the ip.dst column
        of any of the previous frames.
        """
        new_ipsrc = self.new['ip.src'].iloc[0]
        self.new_row['ipsrc_in_ipdst'] = int(self.previous['ip.dst'].str.contains(new_ipsrc).any())
    
    def extract_ipsrc_in_last_ipsrc(self):
        """
        Whether or not the source IP address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['ipsrc_in_last_ipsrc'] = int(self.new['ip.src'].iloc[0] == self.previous['ip.src'].iloc[-1])
    
    def extract_ipsrc_in_last_ipdst(self):
        """
        Whether or not the source IP of the new frame matches the destination IP of
        the last frame of the previous dataframe.
        """
        self.new_row['ipsrc_in_last_ipdst'] = int(self.new['ip.src'].iloc[0] == self.previous['ip.dst'].iloc[-1])
    
    # ip.dst features
    # ==============================================================================================
    def extract_ipdst_in_ipdst(self):
        """
        Whether or not the destination IP of the new frame is in the ip.dst column
        of any of the previous frames.
        """
        new_ipdst = self.new['ip.dst'].iloc[0]
        self.new_row['ipdst_in_ipdst'] = int(self.previous['ip.dst'].str.contains(new_ipdst).any())
    
    def extract_ipdst_in_ipsrc(self):
        """
        Whether or not the destination IP of the new frame is in the ip.src column
        of any of the previous frames.
        """
        new_ipdst = self.new['ip.dst'].iloc[0]
        self.new_row['ipdst_in_ipsrc'] = int(self.previous['ip.src'].str.contains(new_ipdst).any())
    
    def extract_ipdst_in_last_ipdst(self):
        """
        Whether or not the destination IP address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['ipdst_in_last_ipdst'] = int(self.new['ip.dst'].iloc[0] == self.previous['ip.dst'].iloc[-1])
        
    def extract_ipdst_in_last_ipsrc(self):
        """
        Whether or not the destination IP of the new frame matches the source IP of
        the last frame of the previous dataframe.
        """
        self.new_row['ipdst_in_last_ipsrc'] = int(self.new['ip.dst'].iloc[0] == self.previous['ip.src'].iloc[-1])

    # ip.proto features
    # ==============================================================================================
    def extract_ipproto_in_ipproto(self):
        """
        Whether or not the ip protocol of the new frame is found in all the ip protocols
        from the previous dataframe.
        """
        new_ipproto = str(self.new['ip.proto'].iloc[0])
        self.new_row['ipproto_in_ipproto'] = int(new_ipproto in list(map(str, self.previous['ip.proto'].values)))
    
    def extract_ipproto_in_last_ipproto(self):
        """
        Whether or not the ip protocol of the new frame matches with the ip protocol of
        the last frame of the previous dataframe.
        """
        self.new_row['ipproto_in_last_ipproto'] = int(self.new['ip.proto'].iloc[0] == self.previous['ip.proto'].iloc[-1])
    
    def extract_ipproto_fraction_in_ipproto(self):
        """
        Given the ip.proto of the new frame, calculate the percentage of appearances 
        of this protocol in the ip.proto column of the previous dataframe.
        """
        # Get the protocol of the new frame
        new_protocol = self.new['ip.proto'].iloc[0]
        
        # Count how many times this protocol appears in the previous dataframe
        total_protocols = len(self.previous)
        protocol_count = (self.previous['ip.proto'] == new_protocol).sum()
        
        # Calculate the percentage as a fraction of total previous rows
        if total_protocols > 0:
            self.new_row['ipproto_fraction_in_ipproto'] = protocol_count / total_protocols
        else:
            # If there are no previous frames, default the percentage to 0
            self.new_row['ipproto_fraction_in_ipproto'] = 0.0
    
    def extract_current_ipproto_features(self):
        """
        The value of the ip.proto in the new dataframe. It is already extracted in a
        one-hot encoding manner, only defining columns for the most common values.
        """
        new_ipproto = int(self.new['ip.proto'].iloc[0])
        if new_ipproto == 6:
            self.new_row['current_ipproto_TCP'] = 1
            self.new_row['current_ipproto_UDP'] = 0
        elif new_ipproto == 17:
            self.new_row['current_ipproto_TCP'] = 0
            self.new_row['current_ipproto_UDP'] = 1
        else:
            self.new_row['current_ipproto_TCP'] = 0
            self.new_row['current_ipproto_UDP'] = 0
            
    # tcp.flags.str features
    # ==============================================================================================
    def extract_tcpflag_in_tcpflag(self):
        """
        Whether or not the tcpflag of the new frame is found in all the tcpflags
        from the previous dataframe.
        """
        new_tcpflag = self.new['tcp.flags.str'].iloc[0]
        self.new_row['tcpflag_in_tcpflag'] = int(self.previous['tcp.flags.str'].str.contains(new_tcpflag).any())
        
    def extract_tcpflag_in_last50_tcpflag(self):
        """
        Whether or not the tcpflag of the new frame is found in any of the previous 50 of
        the tcpflags from the previous dataframe.
        """
        new_tcpflag = self.new['tcp.flags.str'].iloc[0]
        self.new_row['tcpflag_in_tcpflag_50'] = int(self.previous['tcp.flags.str'].iloc[-50:].str.contains(new_tcpflag).any())
    
    def extract_tcpflag_in_last_tcpflag(self):
        """
        Whether or not the tcpflag of the new frame matches with the tcpflag of
        the last frame of the previous dataframe.
        """
        self.new_row['tcpflag_in_last_tcpflag'] = int(self.new['tcp.flags.str'].iloc[0] == self.previous['tcp.flags.str'].iloc[-1])
    
    def extract_tcpflag_fraction_in_tcpflag_50(self):
        """
        Given the tcpflag of the new frame, calculate the percentage of appearances 
        of this flag in the tcpflag column of the previous dataframe (max last 50).
        """
        # Get the protocol of the new frame
        new_tcpflag = self.new['tcp.flags.str'].iloc[0]
        
        # Count how many times this flag appears in the previous dataframe
        previous = self.previous['tcp.flags.str'].iloc[-50:]
        total_flags = len(previous)
        flag_count = (previous == new_tcpflag).sum()
        
        # Calculate the percentage as a fraction of total previous rows
        if total_flags > 0:
            self.new_row['tcpflag_fraction_in_tcpflag_50'] = flag_count / total_flags
        else:
            # If there are no previous frames, default the percentage to 0
            self.new_row['tcpflag_fraction_in_tcpflag_50'] = 0.0
            
    def extract_tcpflag_fraction_in_tcpflag_ALL(self):
        """
        Given the tcpflag of the new frame, calculate the percentage of appearances 
        of this flag in the tcpflag column of the previous dataframe (max last 50).
        """
        # Get the protocol of the new frame
        new_tcpflag = self.new['tcp.flags.str'].iloc[0]
        
        # Count how many times this flag appears in the previous dataframe
        previous = self.previous['tcp.flags.str']
        total_flags = len(previous)
        flag_count = (previous == new_tcpflag).sum()
        
        # Calculate the percentage as a fraction of total previous rows
        if total_flags > 0:
            self.new_row['tcpflag_fraction_in_tcpflag_ALL'] = flag_count / total_flags
        else:
            # If there are no previous frames, default the percentage to 0
            self.new_row['tcpflag_fraction_in_tcpflag_ALL'] = 0.0
    
    def extract_current_tcpflag_features(self):
        """
        The value of the tcp.flags.str in the new dataframe. It is already extracted in a
        one-hot encoding manner, only defining columns for the most common values.
        """
        new_tcpflag = self.new['tcp.flags.str'].iloc[0]
        if new_tcpflag == "·······A····":
            self.new_row['current_tcpflag_A'] = 1
            self.new_row['current_tcpflag_AP'] = 0
            self.new_row['current_tcpflag_AS'] = 0
            self.new_row['current_tcpflag_S'] = 0
        elif new_tcpflag == "········AP···":
            self.new_row['current_tcpflag_A'] = 0
            self.new_row['current_tcpflag_AP'] = 1
            self.new_row['current_tcpflag_AS'] = 0
            self.new_row['current_tcpflag_S'] = 0
        elif new_tcpflag == "·······A··S·":
            self.new_row['current_tcpflag_A'] = 0
            self.new_row['current_tcpflag_AP'] = 0
            self.new_row['current_tcpflag_AS'] = 1
            self.new_row['current_tcpflag_S'] = 0
        elif new_tcpflag == "··········S·":
            self.new_row['current_tcpflag_A'] = 0
            self.new_row['current_tcpflag_AP'] = 0
            self.new_row['current_tcpflag_AS'] = 0
            self.new_row['current_tcpflag_S'] = 1
        else:
            self.new_row['current_tcpflag_A'] = 0
            self.new_row['current_tcpflag_AP'] = 0
            self.new_row['current_tcpflag_AS'] = 0
            self.new_row['current_tcpflag_S'] = 0

    def extract_current_tcpflag_features_2(self):
        """
        The value of the tcp.flags.str in the new dataframe. It is already extracted in a
        one-hot encoding manner, only defining columns for the most common values. This
        second version uses less columns (S, AS, others)
        """
        new_tcpflag = self.new['tcp.flags.str'].iloc[0]
        if new_tcpflag == "·······A··S·":
            self.new_row['current_tcpflag_AS'] = 1
            self.new_row['current_tcpflag_S'] = 0
        elif new_tcpflag == "··········S·":
            self.new_row['current_tcpflag_AS'] = 0
            self.new_row['current_tcpflag_S'] = 1
        else:
            self.new_row['current_tcpflag_AS'] = 0
            self.new_row['current_tcpflag_S'] = 0
            
    def extract_previous_tcpflag_features_2(self):
        """
        The value of the tcp.flags.str in the last frame of the previous group of frames.
        It is already extracted in a one-hot encoding manner, only defining columns for
        the most common values. This version uses S, AS and others
        """
        previous_tcpflag = self.previous['tcp.flags.str'].iloc[-1]
        if previous_tcpflag == "·······A··S·":
            self.new_row['previous_tcpflag_AS'] = 1
            self.new_row['previous_tcpflag_S'] = 0
        elif previous_tcpflag == "··········S·":
            self.new_row['previous_tcpflag_AS'] = 0
            self.new_row['previous_tcpflag_S'] = 1
        else:
            self.new_row['previous_tcpflag_AS'] = 0
            self.new_row['previous_tcpflag_S'] = 0
            
    # frame.len features
    # ==============================================================================================
    def extract_diff_lenframe_to_last(self):
        """
        The difference in frame length between the new frame and the last frame of the 
        previous dataframe.
        """
        self.new_row['diff_lenframe_to_last'] = self.new['frame.len'].iloc[0] - self.previous['frame.len'].iloc[-1]
    
    def extract_diff_lenframe_to_last_mean_50(self):
        """
        The difference between the frame length of the new frame and the mean of the
        frame length of the previous dataframe (max last 50 frames).
        """    
        previous = self.previous['frame.len'].iloc[-50:]
        self.new_row['diff_lenframe_to_last_mean_50'] = self.new['frame.len'].iloc[0] - previous.mean()
    
    def extract_diff_lenframe_to_last_mean_ALL(self):
        """
        The difference between the frame length of the new frame and the mean of the
        frame length of the previous dataframe (max last 50 frames).
        """    
        previous = self.previous['frame.len']
        self.new_row['diff_lenframe_to_last_meanA_ALL'] = self.new['frame.len'].iloc[0] - previous.mean()
    
    def extract_lenframe_last_variance_50(self):
        """
        The variance of the column frame.len of the max last 50 frames.
        """
        previous = self.previous['frame.len'].iloc[-50:]
        self.new_row['lenframe_last_variance_50'] = np.var(previous)
        
    def extract_lenframe_last_variance_ALL(self):
        """
        The variance of the column frame.len of the max last 50 frames.
        """
        previous = self.previous['frame.len']
        self.new_row['lenframe_last_variance_ALL'] = np.var(previous)
        
    def extract_current_framelen_features(self):
        """
        Classification of the frame.len of the new frame into different lengths cathegories
        observed in the data exploration.
        """
        new_framelen = self.new['frame.len'].iloc[0]
        if new_framelen < 750:
            self.new_row['current_len_0_750'] = 1
            self.new_row['current_len_750_2250'] = 0
            self.new_row['current_len_2250_3750'] = 0
            self.new_row['current_len_3750_5250'] = 0
        elif new_framelen < 2250:
            self.new_row['current_len_0_750'] = 0
            self.new_row['current_len_750_2250'] = 1
            self.new_row['current_len_2250_3750'] = 0
            self.new_row['current_len_3750_5250'] = 0
        elif new_framelen < 3750:
            self.new_row['current_len_0_750'] = 0
            self.new_row['current_len_750_2250'] = 0
            self.new_row['current_len_2250_3750'] = 1
            self.new_row['current_len_3750_5250'] = 0
        elif new_framelen < 5250:
            self.new_row['current_len_0_750'] = 0
            self.new_row['current_len_750_2250'] = 0
            self.new_row['current_len_2250_3750'] = 0
            self.new_row['current_len_3750_5250'] = 1
        else:
            self.new_row['current_len_0_750'] = 0
            self.new_row['current_len_750_2250'] = 0
            self.new_row['current_len_2250_3750'] = 0
            self.new_row['current_len_3750_5250'] = 0
    
    # port.src features
    # ==============================================================================================
    def extract_portsrc_in_portsrc(self):
        """
        Whether or not the source port of the new frame is in the port.src column
        of any of the previous frames.
        """
        new_portsrc = self.new['port.src'].iloc[0]
        self.new_row['portsrc_in_portsrc'] = int(self.previous['port.src'].str.contains(new_portsrc).any())
    
    def extract_portsrc_in_portdst(self):
        """
        Whether or not the source port of the new frame is in the port.dst column
        of any of the previous frames.
        """
        new_portsrc = self.new['port.src'].iloc[0]
        self.new_row['portsrc_in_portdst'] = int(self.previous['port.dst'].str.contains(new_portsrc).any())
    
    def extract_portsrc_in_last_portsrc(self):
        """
        Whether or not the source port address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['portsrc_in_last_portsrc'] = int(self.new['port.src'].iloc[0] == self.previous['port.src'].iloc[-1])
    
    def extract_portsrc_in_last_portdst(self):
        """
        Whether or not the source port of the new frame matches the destination port of
        the last frame of the previous dataframe.
        """
        self.new_row['portsrc_in_last_portdst'] = int(self.new['port.src'].iloc[0] == self.previous['port.dst'].iloc[-1])

    # port.dst features
    # ==============================================================================================
    def extract_portdst_in_portdst(self):
        """
        Whether or not the destination port of the new frame is in the port.dst column
        of any of the previous frames.
        """
        new_portdst = self.new['port.dst'].iloc[0]
        self.new_row['portdst_in_portdst'] = int(self.previous['port.dst'].str.contains(new_portdst).any())
    
    def extract_portdst_in_portsrc(self):
        """
        Whether or not the destination port of the new frame is in the port.src column
        of any of the previous frames.
        """
        new_portdst = self.new['port.dst'].iloc[0]
        self.new_row['portdst_in_portsrc'] = int(self.previous['port.src'].str.contains(new_portdst).any())
    
    def extract_portdst_in_last_portdst(self):
        """
        Whether or not the destination port address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['portdst_in_last_portdst'] = int(self.new['port.dst'].iloc[0] == self.previous['port.dst'].iloc[-1])
        
    def extract_portdst_in_last_portsrc(self):
        """
        Whether or not the destination port of the new frame matches the source port of
        the last frame of the previous dataframe.
        """
        self.new_row['portdst_in_last_portsrc'] = int(self.new['port.dst'].iloc[0] == self.previous['port.src'].iloc[-1])
    
    # endpoint.src features
    # ==============================================================================================
    def extract_endpointsrc_in_endpointsrc(self):
        """
        Whether or not the source endpoint of the new frame is in the endpoint.src column
        of any of the previous frames.
        """
        new_endpointsrc = self.new['endpoint.src'].iloc[0]
        self.new_row['endpointsrc_in_endpointsrc'] = int(self.previous['endpoint.src'].str.contains(new_endpointsrc).any())
    
    def extract_endpointsrc_in_endpointdst(self):
        """
        Whether or not the source endpoint of the new frame is in the endpoint.dst column
        of any of the previous frames.
        """
        new_endpointsrc = self.new['endpoint.src'].iloc[0]
        self.new_row['endpointsrc_in_endpointdst'] = int(self.previous['endpoint.dst'].str.contains(new_endpointsrc).any())
    
    def extract_endpointsrc_in_last_endpointsrc(self):
        """
        Whether or not the source endpoint address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['endpointsrc_in_last_endpointsrc'] = int(self.new['endpoint.src'].iloc[0] == self.previous['endpoint.src'].iloc[-1])
    
    def extract_endpointsrc_in_last_endpointdst(self):
        """
        Whether or not the source endpoint of the new frame matches the destination endpoint of
        the last frame of the previous dataframe.
        """
        self.new_row['endpointsrc_in_last_endpointdst'] = int(self.new['endpoint.src'].iloc[0] == self.previous['endpoint.dst'].iloc[-1])
        
    # endpoint.dst features
    # ==============================================================================================
    def extract_endpointdst_in_endpointdst(self):
        """
        Whether or not the destination endpoint of the new frame is in the endpoint.dst column
        of any of the previous frames.
        """
        new_endpointdst = self.new['endpoint.dst'].iloc[0]
        self.new_row['endpointdst_in_endpointdst'] = int(self.previous['endpoint.dst'].str.contains(new_endpointdst).any())
    
    def extract_endpointdst_in_endpointsrc(self):
        """
        Whether or not the destination endpoint of the new frame is in the endpoint.src column
        of any of the previous frames.
        """
        new_endpointdst = self.new['endpoint.dst'].iloc[0]
        self.new_row['endpointdst_in_endpointsrc'] = int(self.previous['endpoint.src'].str.contains(new_endpointdst).any())
    
    def extract_endpointdst_in_last_endpointdst(self):
        """
        Whether or not the destination endpoint address matches between the new frame and the
        last frame of the previous dataframe.
        """
        self.new_row['endpointdst_in_last_endpointdst'] = int(self.new['endpoint.dst'].iloc[0] == self.previous['endpoint.dst'].iloc[-1])
        
    def extract_endpointdst_in_last_endpointsrc(self):
        """
        Whether or not the destination endpoint of the new frame matches the source endpoint of
        the last frame of the previous dataframe.
        """
        self.new_row['endpointdst_in_last_endpointsrc'] = int(self.new['endpoint.dst'].iloc[0] == self.previous['endpoint.src'].iloc[-1])
    
    # private/public IPs features
    # ==============================================================================================
    def extract_current_last_private_features(self):
        """
        Whether or not the source IP of the new frame is private according to IANA IPv4
        Special-Purpose Address Registry.
        """
        new_ipsrc = self.new['ip.src'].iloc[0]
        
        ip_obj = ipaddress.IPv4Address(new_ipsrc)
        
        if ip_obj.is_private:
            self.new_row['is_current_ipsrc_private'] = 1 # Private IP
        else:
            self.new_row['is_current_ipsrc_private'] = 0 # Public IP
    
        """
        Whether or not the destination IP of the new frame is private according to IANA IPv4
        Special-Purpose Address Registry.
        """
        new_ipdst = self.new['ip.dst'].iloc[0]
        
        ip_obj = ipaddress.IPv4Address(new_ipdst)
        
        if ip_obj.is_private:
            self.new_row['is_current_ipdst_private'] = 1 # Private IP
        else:
            self.new_row['is_current_ipdst_private'] = 0 # Public IP
    
        """
        Whether or not the source IP of the last previous frame is private according to IANA IPv4
        Special-Purpose Address Registry.
        """
        last_previous_ipsrc = self.previous['ip.src'].iloc[-1]
        
        ip_obj = ipaddress.IPv4Address(last_previous_ipsrc)
        
        if ip_obj.is_private:
            self.new_row['is_last_ipsrc_private'] = 1 # Private IP
        else:
            self.new_row['is_last_ipsrc_private'] = 0 # Public IP
    
        """
        Whether or not the destination IP of the last previous frame is private according to IANA IPv4
        Special-Purpose Address Registry.
        """
        last_previous_ipdst = self.previous['ip.dst'].iloc[-1]
        
        ip_obj = ipaddress.IPv4Address(last_previous_ipdst)
        
        if ip_obj.is_private:
            self.new_row['is_last_ipdst_private'] = 1 # Private IP
        else:
            self.new_row['is_last_ipdst_private'] = 0 # Public IP

        # Creating the final features: 
        """
        Whether source/destination IP of the new frame and source/destination IP of the last
        frame of the previous frames are both private or both public.
        """
        self.new_row['current_ipsrc_last_ipsrc_private'] = self.new_row['is_current_ipsrc_private'] & self.new_row['is_last_ipsrc_private']
        self.new_row['current_ipsrc_last_ipsrc_public'] = ~(self.new_row['is_current_ipsrc_private'] | self.new_row['is_last_ipsrc_private']) & 1
        
        self.new_row['current_ipdst_last_ipdst_private'] = self.new_row['is_current_ipdst_private'] & self.new_row['is_last_ipdst_private']
        self.new_row['current_ipdst_last_ipdst_public'] = ~(self.new_row['is_current_ipdst_private'] | self.new_row['is_last_ipdst_private']) & 1
        
        self.new_row['current_ipsrc_last_ipdst_private'] = self.new_row['is_current_ipsrc_private'] & self.new_row['is_last_ipdst_private']
        self.new_row['current_ipsrc_last_ipdst_public'] = ~(self.new_row['is_current_ipsrc_private'] | self.new_row['is_last_ipdst_private']) & 1
        
        self.new_row['current_ipdst_last_ipsrc_private'] = self.new_row['is_current_ipdst_private'] & self.new_row['is_last_ipsrc_private']
        self.new_row['current_ipdst_last_ipsrc_public'] = ~(self.new_row['is_current_ipdst_private'] | self.new_row['is_last_ipsrc_private']) & 1

        # Delete auxiliary features
        keys_to_delete = ['is_current_ipsrc_private', 'is_last_ipsrc_private', 'is_current_ipdst_private', 'is_last_ipdst_private']
        for key in keys_to_delete:
            self.new_row.pop(key, None)
    
    # Other features
    # ==============================================================================================
    def extract_length_previous(self):
        """
        How many rows long was the previous dataset.
        """
        self.new_row['length_previous'] = self.previous.shape[0]
    
    def get_new_row(self):
        return self.new_row
