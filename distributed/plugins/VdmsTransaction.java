import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class VdmsTransaction
{
    byte size[];
    byte buffer[];
    int id;
    long timestamp;

    
    public VdmsTransaction(byte[] nSize, byte[] nBuffer)
    {
        size = nSize;
        buffer = nBuffer;
        id = -1;
        timestamp = 0;
    }
    
    public VdmsTransaction(int nSize, byte[] nBuffer)
    {
        size = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(nSize).array();
        buffer = nBuffer;
        id = -1;
        timestamp = 0;
    }

    public byte[] GetSize()
    {
    	return size;
    }

    public byte[] GetBuffer()
    {
	    return buffer;
    }

    public int GetId()
    {
        return id;
    }

    public void SetId(int nId)
    {
        id = nId;
    }

    public long GetTimestamp()
    {
        return timestamp;
    }

    public void SetTimestamp(long nTimestamp)
    {
        timestamp = nTimestamp;
    }

}
