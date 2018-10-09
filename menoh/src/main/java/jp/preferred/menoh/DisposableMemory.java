package jp.preferred.menoh;

import com.sun.jna.Memory;

/**
 * A {@link Memory} class which can {@link #dispose()} the underlying native memory explicitly.
 */
class DisposableMemory extends Memory implements AutoCloseable {
    public DisposableMemory(long size) {
        super(size);
    }

    @Override
    public synchronized void dispose() {
        super.dispose();
    }

    @Override
    public void close() {
        dispose();
    }

    public boolean isDisposed() {
        return this.peer == 0;
    }
}
